"""
probe_bowl_attraction.py

Diagnostic probe, not a `run_eval.sh --split`: real action rollouts (unlike
`probe_bowl_pointing*.py`, which hit a dead end -- see benchmark_split_result.md Sec.8 -- because
OpenVLA's free-text/logit channel isn't language-responsive), instrumented with per-step
end-effector-to-bowl distance so a rollout's *failure mode* can be read off behaviorally.

Motivation. Split 1 showed distractor-mention prompts (`negative_contrast`/`positive_contrast`)
collapse task success (e.g. task 5 "on the ramekin": 94% -> 4%/2%), while a genuinely separate VLM
(Qwen2-VL) resolves the identical referring expression at 70% accuracy (benchmark_split_result.md
Sec.8.2). But Split 4b showed a prompt that rewords *only* the target -- no second bowl mentioned
at all, e.g. `target_cue_landmark`'s "next to the ramekin" for the same task 5 -- collapses success
just as hard (94% -> 10%), which argues the mechanism is template mismatch (the LoRA fine-tune only
ever saw 10 fixed target-only sentence templates), not confusion about *which* bowl is meant. This
script tests that directly: on task 5's scene (2 bowls always physically present, distractor never
moved), does the end effector get pulled toward the *distractor* bowl under `negative_contrast`
(supporting "language pulls it to the wrong object"), or does it fail to lock onto *either* bowl
coherently even under `target_cue_landmark` (supporting "reword collapses reaching before target
selection is even in play", consistent with Split 4b's mechanism)?

Not wired into eval_registry.py -- this instruments a handful of real rollouts per condition, not a
benchmark split.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np

sys.path.append("../..")
from libero.libero import benchmark

from experiments.robot.libero.eval_registry import CONDITIONS
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class ProbeConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    task_id: int = 5                                  # "on the ramekin" -- largest single-task
                                                        # negative_contrast/target_cue_landmark collapse
    conditions: str = "default,negative_contrast,target_cue_landmark"  # comma-separated
    unnorm_key: Optional[str] = None

    num_trials: int = 15                               # rollouts per condition
    num_steps_wait: int = 10
    near_thresh_m: float = 0.08                        # eef-to-bowl-center distance counted as
                                                        # "reaching for" that bowl (~bowl radius
                                                        # 0.0575m + gripper-finger clearance)
    seed: int = 7
    save_videos: bool = True
    local_log_dir: str = "./experiments/logs/probe_bowl_attraction"
    run_id_note: Optional[str] = None
    # fmt: on


@draccus.wrap()
def probe(cfg: ProbeConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must be set!"
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.unnorm_key or cfg.task_suite_name

    model = get_model(cfg)
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"unnorm_key {cfg.unnorm_key} not in model.norm_stats"
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    resize_size = get_image_resize_size(cfg)

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_key = f"{cfg.task_suite_name}--t{cfg.task_id}"
    if cfg.run_id_note:
        run_key += f"--{cfg.run_id_note}"
    out_path = os.path.join(cfg.local_log_dir, f"{run_key}--{DATE_TIME}.jsonl")
    out_file = open(out_path, "a")
    print(f"Writing per-episode attraction records to {out_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    init_states = task_suite.get_task_init_states(cfg.task_id)

    conditions = cfg.conditions.split(",")
    summary = {}

    for condition in conditions:
        assert condition in CONDITIONS, f"Unknown condition '{condition}'. Available: {sorted(CONDITIONS)}"
        instruction_map = CONDITIONS[condition]
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        if instruction_map is not None:
            assert task.name in instruction_map, f"No '{condition}' instruction for task '{task.name}'"
            task_description = instruction_map[task.name]
        print(f"\n=== condition={condition!r}  instruction={task_description!r} ===")

        base = env.env  # underlying robosuite env -- same access pattern as bowl_pointing_common.py
        bowl_names = sorted(n for n in base.obj_body_id if n.startswith("akita_black_bowl"))
        assert "akita_black_bowl_1" in bowl_names, f"task {task.name} has no akita_black_bowl_1 target"
        bowl_body_ids = {n: base.obj_body_id[n] for n in bowl_names}

        records = []
        for episode_idx in range(cfg.num_trials):
            env.reset()
            obs = env.set_init_state(init_states[episode_idx])

            t = 0
            max_steps = 220  # matches run_libero_eval.py's libero_spatial* cap
            replay_images = []
            # per-bowl running min distance (3D eef-to-bowl-center) and the step it was first
            # observed inside cfg.near_thresh_m ("reached for" that bowl)
            min_dist = {n: float("inf") for n in bowl_names}
            first_near_step = {n: None for n in bowl_names}
            done = False

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    eef_pos = np.array(obs["robot0_eef_pos"])
                    for n, bid in bowl_body_ids.items():
                        bowl_pos = np.array(base.sim.data.body_xpos[bid])
                        d = float(np.linalg.norm(eef_pos - bowl_pos))
                        if d < min_dist[n]:
                            min_dist[n] = d
                        if d < cfg.near_thresh_m and first_near_step[n] is None:
                            first_near_step[n] = t

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    action = get_action(cfg, model, observation, task_description, processor=processor)
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        break
                    t += 1
                except Exception as e:
                    print(f"Caught exception: {e}")
                    break

            # First bowl actually reached for, by step order (None if the gripper never got
            # within near_thresh_m of any bowl the whole episode).
            reached = [(step, n) for n, step in first_near_step.items() if step is not None]
            reached.sort()
            first_bowl = reached[0][1] if reached else None
            first_bowl_is_target = (first_bowl == "akita_black_bowl_1") if first_bowl else None

            rec = {
                "task_id": cfg.task_id,
                "task_name": task.name,
                "condition": condition,
                "instruction": task_description,
                "episode_idx": episode_idx,
                "success": bool(done),
                "min_dist_by_bowl": min_dist,
                "first_near_step_by_bowl": first_near_step,
                "first_bowl_approached": first_bowl,
                "first_bowl_is_target": first_bowl_is_target,
            }
            records.append(rec)
            out_file.write(json.dumps(rec) + "\n")
            out_file.flush()
            print(
                f"  ep {episode_idx}: success={rec['success']} first_bowl={first_bowl} "
                f"min_dist_target={min_dist['akita_black_bowl_1']:.3f}"
            )

            if cfg.save_videos:
                save_rollout_video(
                    replay_images,
                    episode_idx,
                    success=done,
                    task_description=f"{condition}--{task_description}",
                    log_file=None,
                )

        env.close()

        n = len(records)
        n_success = sum(r["success"] for r in records)
        n_target_first = sum(r["first_bowl_is_target"] is True for r in records)
        n_distractor_first = sum(r["first_bowl_is_target"] is False for r in records)
        n_neither = sum(r["first_bowl_approached"] is None for r in records)
        summary[condition] = {
            "n": n,
            "success_rate": n_success / n,
            "target_approached_first_rate": n_target_first / n,
            "distractor_approached_first_rate": n_distractor_first / n,
            "neither_approached_rate": n_neither / n,
            "mean_min_dist_target": float(np.mean([r["min_dist_by_bowl"]["akita_black_bowl_1"] for r in records])),
        }

    print("\n=== Summary ===")
    for condition, s in summary.items():
        print(f"{condition}: {json.dumps(s, indent=2)}")

    summary_path = os.path.join(cfg.local_log_dir, f"{run_key}--{DATE_TIME}--summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    probe()
