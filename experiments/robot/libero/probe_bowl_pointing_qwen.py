"""
probe_bowl_pointing_qwen.py

Runs the bowl-pointing VQA probe (see probe_bowl_pointing.py's module docstring) against a
general-purpose VLM never trained on OpenVLA's action-only template, instead of the OpenVLA
checkpoint itself. Motivation: probe_bowl_pointing.py established (see benchmark_split_result.md
Sec.8) that OpenVLA -- base or LIBERO-finetuned -- has no text-output channel responsive to language
input at all, so grounding can't be separated from action decoding on that model family. This script
answers a narrower, still-useful question instead: is the referring expression in the
distractor-mention conditions resolvable IN PRINCIPLE by a capable VLM, or is the scene/language
itself ambiguous regardless of which model looks at it?

Uses the same rendered/annotated images and ground truth as probe_bowl_pointing.py (via
bowl_pointing_common.py) so results are directly comparable in setup, just not in what they imply
about THIS project's checkpoint.

Sampling (2026-09-04). Every prior run of this script (benchmark_split_result.md Sec.8.1-8.4) drew
exactly one greedy (`do_sample=False`) generation per query, so a condition's reported accuracy was
a single point estimate over already-small n (10 or 40 pooled) -- no sense of how much of that
number is model signal vs. one unlucky/lucky decode. This version draws `num_samples` sampled
generations per query (`do_sample=True`, `temperature=cfg.temperature`, via `num_return_sequences`
in one `generate()` call) and reports, per query, the fraction of samples that land on the correct
bowl (`sample_accuracy`) and the majority-vote answer -- the former is the trend metric this change
was for; the latter is the closest analog to the old greedy-only numbers, for continuity. Set
`--num_samples 1 --temperature 0` to reproduce the exact old greedy behavior.

Dependency note: unlike probe_bowl_pointing.py, this needs `transformers>=4.49` and `qwen-vl-utils`
for Qwen2-VL support -- newer than the eval image's pinned `transformers==4.40.1`. Install ephemerally
inside a container rather than rebuilding the shared eval image:
    pip install -U "transformers>=4.49" accelerate qwen-vl-utils
bowl_pointing_common.py deliberately has no dependency on experiments.robot.openvla_utils or any
transformers-version-sensitive OpenVLA code, so this upgrade is safe to do in the same container.

Usage:
    python experiments/robot/libero/probe_bowl_pointing_qwen.py \
        --conditions negative_contrast,positive_contrast,hardneg \
        --task_ids 0,1 \                # smoke test a couple of tasks before running the full 10
        --num_samples 10 --temperature 0.7   # sampled trend, not just one greedy decode (default)
"""

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import draccus
import numpy as np
import torch
from libero.libero import benchmark
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

sys.path.append("../..")
from experiments.robot.libero.bowl_pointing_common import CONDITION_SUITES, parse_answer, render_and_annotate

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        "qwen_vl_utils is required. Install with: pip install -U 'transformers>=4.49' accelerate qwen-vl-utils"
    ) from e


@dataclass
class QwenProbeConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"

    conditions: str = "default,negative_contrast,positive_contrast,hardneg"  # comma-separated, keys of CONDITION_SUITES
    task_ids: Optional[str] = None  # comma-separated task id filter, default: all 10

    resolution: int = 256      # render resolution, matches get_libero_env's eval default
    resize_size: int = 224     # model input size, matches get_image_resize_size()
    num_steps_wait: int = 10   # settle steps before capture, matches run_libero_eval.py's default
    max_new_tokens: int = 128  # Qwen isn't action-token constrained, allow room for a short answer
    seed: int = 7

    num_samples: int = 10      # sampled generations per query (trend, not one greedy decode)
    temperature: float = 0.7   # set 0 (with num_samples=1) to reproduce the old greedy behavior

    local_log_dir: str = "./experiments/logs/probe_bowl_pointing_qwen"
    fig_dir: str = "./experiments/figures/probe_bowl_pointing"  # shared with probe_bowl_pointing.py


def query_qwen(model, processor, annotated_image, instruction, max_new_tokens, num_samples, temperature):
    """Returns a list of `num_samples` decoded answers for the same (image, question) pair.

    `temperature <= 0` falls back to a single greedy (`do_sample=False`) generation, matching this
    script's original behavior, regardless of `num_samples` (greedy has no sample-to-sample
    variation, so requesting more than one would just duplicate it).
    """
    question = (
        f"The black bowls in this image are marked with numbers. Which numbered bowl should you "
        f"{instruction} Answer with just the number, then briefly explain why."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": annotated_image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    do_sample = temperature > 0
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(temperature=temperature, num_return_sequences=num_samples)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inputs.input_ids[0]):] for out in generated_ids]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded if do_sample else decoded[:1]


@draccus.wrap()
def probe(cfg: QwenProbeConfig) -> None:
    conditions = cfg.conditions.split(",")
    for c in conditions:
        assert c in CONDITION_SUITES, f"Unknown condition '{c}'. Available: {sorted(CONDITION_SUITES)}"
    task_id_filter = {int(t) for t in cfg.task_ids.split(",")} if cfg.task_ids else None

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)

    print(f"[*] Loading {cfg.model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    rng = np.random.RandomState(cfg.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    render_cache = {}  # (suite_name, task_id) -> (annotated, bowl_to_number, target_number, raw)

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

            raw_texts = query_qwen(
                model, processor, annotated, instruction, cfg.max_new_tokens, cfg.num_samples, cfg.temperature
            )
            valid_numbers = set(bowl_to_number.values())
            parsed_answers = [parse_answer(t, valid_numbers) for t in raw_texts]
            n_correct = sum(p == target_number for p in parsed_answers)
            sample_accuracy = n_correct / len(parsed_answers)

            answer_counts = Counter(parsed_answers)
            majority_answer, majority_count = answer_counts.most_common(1)[0]
            majority_correct = majority_answer == target_number

            fig_path = os.path.join(cfg.fig_dir, f"{suite_name}--{condition}--t{task_id}.png")
            annotated.save(fig_path)

            record = {
                "model_id": cfg.model_id,
                "condition": condition,
                "task_suite_name": suite_name,
                "task_id": task_id,
                "task_name": task.name,
                "instruction": instruction,
                "bowl_to_number": bowl_to_number,
                "target_number": target_number,
                "num_samples": len(raw_texts),
                "temperature": cfg.temperature,
                "raw_model_outputs": raw_texts,
                "parsed_answers": parsed_answers,
                "sample_accuracy": sample_accuracy,
                "answer_counts": {str(k): v for k, v in answer_counts.items()},
                "majority_answer": majority_answer,
                "majority_correct": majority_correct,
                "annotated_image": fig_path,
            }
            results.append(record)
            print(
                f"[{condition} t{task_id}] target={target_number} sample_acc={sample_accuracy:.2f} "
                f"({n_correct}/{len(parsed_answers)}) majority={majority_answer} "
                f"majority_correct={majority_correct} answers={dict(answer_counts)}"
            )

    results_path = os.path.join(cfg.local_log_dir, "probe_bowl_pointing_qwen.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nStructured results: {results_path}")
    print(f"Annotated images: {cfg.fig_dir}")
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        if not rows:
            continue
        mean_sample_acc = sum(r["sample_accuracy"] for r in rows) / len(rows)
        n_majority_correct = sum(r["majority_correct"] for r in rows)
        print(
            f"{condition}: sample_accuracy(mean over {rows[0]['num_samples']} samples/query)="
            f"{mean_sample_acc*100:.1f}%  |  majority-vote={n_majority_correct}/{len(rows)} "
            f"({n_majority_correct/len(rows)*100:.1f}%)"
        )


if __name__ == "__main__":
    probe()
