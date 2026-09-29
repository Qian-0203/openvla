"""
probe_bowl_pointing.py

Diagnostic VQA probe, NOT an action-rollout eval -- not wired into eval_registry.py/SPLITS.

This project's distractor-mention conditions (negative_contrast, positive_contrast, hardneg) crash
the fine-tuned checkpoint's task success rate (see vla_ws/docs/benchmark_split_result.md Sec.6 findings
1, 5, 9-11), but end-to-end success can't tell you WHY: is vision-language grounding actually broken
once a second referent is mentioned, or is grounding fine and only the action-decoding head falls
apart on this out-of-distribution phrasing?

This script isolates grounding from action generation. For each task, it:
  1. Renders the exact episode-0 init state the real eval would see (same settle-step count/dummy
     action as run_libero_eval.py), through the same 180-degree flip + resize_image() pipeline
     get_libero_image() uses -- so the geometry matches training preprocessing.
  2. Reads each black bowl's true 3D position (`sim.data.body_xpos`) and projects it into the
     rendered frame via robosuite's camera_utils, then overlays a randomly-shuffled number on each
     bowl (Set-of-Mark-style annotation). Ground truth = the number landed on `akita_black_bowl_1`,
     which every task's BDDL goal predicate names as the target (see instructions.py's docstring).
  3. Calls the SAME checkpoint's `.generate()` directly (bypassing `predict_action()`'s
     action-token-only decoding) with the condition's exact failing instruction text, reformatted
     as "which numbered bowl...", and asks it to answer with a number instead of acting.
  4. Scores whether the parsed answer matches the target bowl's number, and saves the annotated
     image + raw generated text for every task (the raw text matters as much as the score -- if the
     checkpoint can't produce a clean answer at all, that itself is informative, since the LoRA head
     was trained almost exclusively on action tokens and free-text quality was untested going in).

Usage (inside the project's docker image, matching run_eval.sh's invocation pattern):
    python experiments/robot/libero/probe_bowl_pointing.py \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --conditions negative_contrast,positive_contrast,hardneg \
        --task_ids 0,1   # smoke test a couple of tasks before running the full 10
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import draccus
import numpy as np
import torch
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.bowl_pointing_common import CONDITION_SUITES, parse_answer, render_and_annotate
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla


@dataclass
class ProbeConfig:
    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    conditions: str = "default,negative_contrast,positive_contrast,hardneg"  # comma-separated, keys of CONDITION_SUITES
    task_ids: Optional[str] = None  # comma-separated task id filter, default: all 10

    resolution: int = 256      # render resolution, matches get_libero_env's eval default
    resize_size: int = 224     # model input size, matches get_image_resize_size()
    num_steps_wait: int = 10   # settle steps before capture, matches run_libero_eval.py's default
    max_new_tokens: int = 16
    seed: int = 7

    local_log_dir: str = "./experiments/logs/probe_bowl_pointing"
    fig_dir: str = "./experiments/figures/probe_bowl_pointing"


def query_model(vla, processor, annotated_image, instruction, max_new_tokens):
    """Calls the checkpoint's generate() directly (bypassing predict_action's action-only
    decoding) to get a free-text answer. Returns the raw decoded completion text."""
    prompt = (
        f"In: The black bowls in this image are marked with numbers. Which numbered bowl should "
        f"you {instruction} Answer with just the number.\nOut:"
    )
    inputs = processor(prompt, annotated_image).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    # Match predict_action's training-time formatting: an empty token (29871) right after "Out:".
    if not torch.all(input_ids[:, -1] == 29871):
        pad = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat((input_ids, pad), dim=1)
    gen_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}

    with torch.no_grad():
        generated_ids = vla.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, **gen_kwargs)
    completion_ids = generated_ids[0, input_ids.shape[1]:]
    return processor.tokenizer.decode(completion_ids, skip_special_tokens=True)


@draccus.wrap()
def probe(cfg: ProbeConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    conditions = cfg.conditions.split(",")
    for c in conditions:
        assert c in CONDITION_SUITES, f"Unknown condition '{c}'. Available: {sorted(CONDITION_SUITES)}"
    task_id_filter = {int(t) for t in cfg.task_ids.split(",")} if cfg.task_ids else None

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)

    print("[*] Loading model...")
    vla = get_vla(cfg)
    processor = get_processor(cfg)

    rng = np.random.RandomState(cfg.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    render_cache = {}  # (suite_name, task_id) -> (annotated, bowl_to_number, target_number)

    results = []
    for condition in conditions:
        suite_name, instruction_dict = CONDITION_SUITES[condition]
        task_suite = benchmark_dict[suite_name]()
        task_ids = [t for t in range(task_suite.n_tasks) if task_id_filter is None or t in task_id_filter]

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            cache_key = (suite_name, task_id)
            if cache_key not in render_cache:
                print(f"[*] Rendering {suite_name} task {task_id} ({task.name})...")
                render_cache[cache_key] = render_and_annotate(
                    task_suite, task_id, cfg.resolution, cfg.resize_size, cfg.num_steps_wait, rng
                )
            annotated, bowl_to_number, target_number, _, default_description = render_cache[cache_key]

            if instruction_dict is None:
                instruction = default_description  # "default" condition: LIBERO's own task language
            else:
                assert task.name in instruction_dict, f"No '{condition}' instruction for task '{task.name}'"
                instruction = instruction_dict[task.name]

            raw_text = query_model(vla, processor, annotated, instruction, cfg.max_new_tokens)
            parsed = parse_answer(raw_text, set(bowl_to_number.values()))
            correct = parsed == target_number

            fig_path = os.path.join(cfg.fig_dir, f"{suite_name}--{condition}--t{task_id}.png")
            annotated.save(fig_path)

            record = {
                "condition": condition,
                "task_suite_name": suite_name,
                "task_id": task_id,
                "task_name": task.name,
                "instruction": instruction,
                "bowl_to_number": bowl_to_number,
                "target_number": target_number,
                "raw_model_output": raw_text,
                "parsed_answer": parsed,
                "correct": correct,
                "annotated_image": fig_path,
            }
            results.append(record)
            print(
                f"[{condition} t{task_id}] target={target_number} parsed={parsed} "
                f"correct={correct} raw={raw_text!r}"
            )

    results_path = os.path.join(cfg.local_log_dir, "probe_bowl_pointing.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nStructured results: {results_path}")
    print(f"Annotated images: {cfg.fig_dir}")
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        acc = sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")
        print(f"{condition}: {sum(r['correct'] for r in rows)}/{len(rows)} correct ({acc*100:.1f}%)")


if __name__ == "__main__":
    probe()
