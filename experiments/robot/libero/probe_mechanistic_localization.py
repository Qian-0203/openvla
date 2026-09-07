"""
probe_mechanistic_localization.py

Validated: success rates reproduce the real eval (task 5: 93.3%/0% here vs. 92-94%/2-4% real, see
benchmark_split_result.md Sec.8.10) after the center-crop and teacher-forced-diagnostics fixes below.
Run four times (Sec.8.9-8.11); see get_action_with_diagnostics's own docstring for the sdpa/eager fix.

Diagnostic probe for benchmark_split_result.md Sec.8.6 gap #3 ("no mechanistic localization").
Sec.8.5's probe_bowl_attraction.py shows *what the arm does* (never commits to a target, or
approaches correctly then still fails) but not *where in the network* it goes wrong. This script
adds two cheap, no-extra-training diagnostics at every action-chunk prediction:

  1. Logit lens: apply the model's own (norm + lm_head) to the *intermediate*-layer hidden state
     feeding each of the 7 action-token predictions, and track the layer at which the eventually-
     chosen token first becomes the layer-wise argmax ("resolution layer"). A late/never-resolving
     pattern during "never commits" failures would point at the LLM backbone /action decoding, not
     the vision encoder.
  2. Attention mass: what fraction of each action-token's attention (averaged over heads) lands on
     the vision-patch span vs. the text-instruction span vs. previously-generated action tokens.
     Diffuse/low vision-attention during a failure would support "not looking at the right bowl";
     high vision-attention with a still-wrong action would push the explanation downstream instead.

Both come for free from a standard HF `.generate(..., output_attentions=True,
output_hidden_states=True, return_dict_in_generate=True)` call -- OpenVLAForActionPrediction only
overrides `prepare_inputs_for_generation`, not `generate` itself (modeling_prismatic.py). We can't
use `vla.predict_action()` here: it assumes `self.generate(...)` returns a raw tensor, which stops
being true once `return_dict_in_generate=True` is passed through its **kwargs. So this script calls
`vla.generate()` directly and replicates predict_action's bin-decoding tail itself.

Caveats (remaining, as of the Sec.8.11 dist_full run):
  - [RESOLVED] Logit lens applies `language_model.model.norm` before `lm_head` (standard Llama
    pre-head norm). Attribute path confirmed correct for this checkpoint's `language_model` class --
    resolution-layer numbers have been consistent (0-32 range, gripper resolving early) across four runs.
  - [RESOLVED 2026-09-06] center_crop preprocessing was previously accepted as a cfg flag but never
    applied here (see benchmark_split_result.md Sec.8.9) -- the 2026-09-04 run's 0% success in both
    conditions (vs. the real eval's 92-94%/2-4% for this task) is attributed to this gap. Now applied
    identically to openvla_utils.get_vla_action (same crop_scale=0.9, same crop_and_resize call).
  - Attentions/hidden-states are NOT serialized to the jsonl (not JSON-safe, and 32-layer x
    per-head tensors add up fast over a 220-step rollout) -- only scalar summaries are written.
    Pass --save_raw_tensors True to additionally dump one .pt per instrumented step for deeper
    offline analysis (do this on a handful of steps/episodes only).
  - Doubling compute per step (diagnostic generate() call runs the same decode the rollout already
    needs) is avoided by using this call's own output to step the env -- but output_attentions=True
    does add memory/latency overhead per step, hence max_env_steps_to_instrument below.
  - Only instruments a prefix of each episode by default (`max_env_steps_to_instrument`), since
    Sec.8.5's "never commits" pattern is about early behavior and full-episode instrumentation is
    expensive; widen once the early-step signal is validated.

Not wired into eval_registry.py -- diagnostic probe, not a benchmark split.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tensorflow as tf
import torch

sys.path.append("../..")
from libero.libero import benchmark

from experiments.robot.libero.eval_registry import CONDITIONS
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import DEVICE, OPENVLA_V01_SYSTEM_PROMPT, crop_and_resize, get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    ACTION_DIM,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from PIL import Image


@dataclass
class ProbeConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    task_id: int = 5                                   # matches probe_bowl_attraction.py's default
    conditions: str = "default,negative_contrast"       # comma-separated; keep short, this is expensive
    unnorm_key: Optional[str] = None

    num_trials: int = 3                                 # small -- this is a compute-heavy first pass
    num_steps_wait: int = 10
    max_env_steps_to_instrument: int = 30                # only the first N post-warmup env steps get
                                                          # full attention/hidden-state extraction
    save_raw_tensors: bool = False                       # if True, also dump one .pt per instrumented
                                                          # step (attentions + hidden_states) for offline use
    seed: int = 7
    local_log_dir: str = "./experiments/logs/probe_mechanistic_localization"
    run_id_note: Optional[str] = None
    # fmt: on


def _build_prompt(base_vla_name: str, task_label: str) -> str:
    """Mirrors openvla_utils.get_vla_action's prompt construction exactly."""
    if "openvla-v01" in base_vla_name:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def _get_num_vision_patches(vla, inputs) -> int:
    """
    One-off forward pass to read off how many patch-embedding tokens the projector emits, so we
    know which span of `attentions`' key dimension is vision vs. text. modeling_prismatic.py's
    forward() builds `multimodal_embeddings = [BOS, projected_patch_embeddings, rest-of-text]`,
    so vision occupies input positions [1, 1 + num_patches).
    """
    with torch.no_grad():
        out = vla(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            output_projector_features=True,
            return_dict=True,
        )
    return out.projector_features.shape[1]


def _decode_action_tokens(vla, generated_token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
    """Replicates OpenVLAForActionPrediction.predict_action's tail (bin decode + unnormalize)."""
    discretized_actions = vla.vocab_size - generated_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized_actions = vla.bin_centers[discretized_actions]

    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def get_action_with_diagnostics(
    vla, processor, base_vla_name: str, obs, task_label: str, unnorm_key: str,
    num_vision_patches: int, center_crop: bool = False,
):
    """
    Like openvla_utils.get_vla_action, but additionally returns a diagnostics dict.

    [2026-09-06] Two-pass, teacher-forced design -- replaces an earlier version that called
    vla.generate(..., output_attentions=True, ...) directly and used ITS OWN argmax as the executed
    action. A controlled single-frame comparison found that doing so changes the actual action: this
    checkpoint's action-token argmax runs in bf16, and requesting output_attentions=True forces a
    fallback from `sdpa` to the numerically-different `eager` attention path (HF's own warning: sdpa
    doesn't support output_attentions). On the very first prediction of a fresh episode, this flipped
    the argmax on 4 of 7 action dims (all continuous ones; the gripper dim, which resolves far earlier
    -- see resolution_layer_frac's dim-6 outlier -- was unaffected both times, consistent with a small
    top-1/runner-up logit margin being what makes a dimension sensitive to this kind of numerical
    perturbation). That made every prior rollout's *behavior* (not just its diagnostics) different from
    what predict_action()/get_vla_action() would have actually done, on top of the separately-fixed
    missing-center-crop bug -- see benchmark_split_result.md Sec.8.9.

    Fix: run the fast path first (plain generate(), whatever attn_implementation the model was loaded
    with -- matches predict_action() exactly) to decide the actual executed action, then run a SEPARATE
    non-incremental forward pass, teacher-forced on those same tokens, with output_attentions=True
    purely to read off hidden_states/attentions. Forcing eager on that second pass is fine now: its own
    argmax is recorded (`diag_argmax_matches_executed`) but never used to pick an action.
    """
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    # Mirrors openvla_utils.get_vla_action's center-crop step exactly (same crop_scale=0.9).
    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy()).convert("RGB")

    prompt = _build_prompt(base_vla_name, task_label)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            [input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)], dim=1
        )
        inputs["input_ids"] = input_ids
    prompt_len = input_ids.shape[1]

    # --- Pass 1 (fast path): decides the actual executed action, identical to predict_action(). ---
    with torch.no_grad():
        fast_ids = vla.generate(**inputs, max_new_tokens=ACTION_DIM, do_sample=False)
    generated_token_ids = fast_ids[0, -ACTION_DIM:].cpu().numpy()
    action = _decode_action_tokens(vla, generated_token_ids, unnorm_key)

    # --- Pass 2 (diagnostics only): one non-incremental, teacher-forced forward over prompt + the
    # exact tokens pass 1 produced. Mathematically equivalent hidden-states/attentions to an
    # incremental generate() over the same tokens (modeling_prismatic.py's forward() splices patch
    # embeddings after position 0 for ANY input_ids.shape[1] != 1 with pixel_values given -- the same
    # splice a cached, per-token generate() call does once at prefill and then reuses via KV-cache).
    gen_ids_tensor = torch.tensor(generated_token_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
    full_input_ids = torch.cat([input_ids, gen_ids_tensor], dim=1)
    full_attention_mask = torch.ones_like(full_input_ids)
    with torch.no_grad():
        diag_out = vla(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    layer_hidden_states = diag_out.hidden_states  # tuple: (embeddings, layer_1, ..., layer_L), fixed seq len
    layer_attns_all = diag_out.attentions
    num_layers = len(layer_hidden_states) - 1

    lm_head = vla.get_output_embeddings()
    final_norm = getattr(getattr(vla.language_model, "model", None), "norm", None)  # Llama-style
    vision_lo, vision_hi = 1, 1 + num_vision_patches

    per_token_diag = []
    for step in range(ACTION_DIM):
        chosen_id = int(generated_token_ids[step])
        # Position (in the patch-spliced sequence forward() builds) whose hidden state/attention row
        # fed this step's prediction: text position (prompt_len - 1 + step), shifted by num_vision_patches
        # because forward() inserts the patches right after position 0 (see modeling_prismatic.py's
        # `multimodal_embeddings = cat([bos, projected_patches, rest_of_text])`).
        row = prompt_len - 1 + step + num_vision_patches

        resolution_layer = None
        final_layer_top1 = None
        layer_top1_ids = []
        for layer_idx, h in enumerate(layer_hidden_states):
            h_row = h[:, row, :]
            if final_norm is not None:
                h_row = final_norm(h_row)
            logits = lm_head(h_row)
            top1_id = int(torch.argmax(logits, dim=-1).item())
            layer_top1_ids.append(top1_id)
            if layer_idx == num_layers:
                final_layer_top1 = top1_id
        for layer_idx, top1_id in enumerate(layer_top1_ids):
            if top1_id == final_layer_top1 and resolution_layer is None:
                resolution_layer = layer_idx

        # Attention mass onto the vision-patch span, averaged over heads, per LLM layer, for the row
        # that produced this step's token. Full (non-cached) forward pass -> every layer's attention
        # tensor already covers the whole sequence, causally masked, so the same fixed `row` index into
        # a fixed-length kv dimension works for every layer.
        vision_attn_by_layer = []
        for layer_attn in layer_attns_all:
            attn_row = layer_attn[0, :, row, :]  # [num_heads, kv_len]
            attn_row = attn_row.mean(dim=0)  # average over heads
            kv_len = attn_row.shape[0]
            vision_mass = attn_row[vision_lo : min(vision_hi, kv_len)].sum().item()
            vision_attn_by_layer.append(float(vision_mass))  # attentions already sum to 1 over kv_len

        per_token_diag.append({
            "action_dim": step,
            "chosen_token_id": chosen_id,                  # from the fast/executed path (pass 1)
            "diag_argmax_matches_executed": final_layer_top1 == chosen_id,  # sdpa-vs-eager divergence
            "resolution_layer": resolution_layer,     # None => never matched final layer (shouldn't happen)
            "resolution_layer_frac": (resolution_layer / num_layers) if resolution_layer is not None else None,
            "vision_attn_by_layer": vision_attn_by_layer,
            "vision_attn_last_layer": vision_attn_by_layer[-1],
            "vision_attn_mean_layer": float(np.mean(vision_attn_by_layer)),
        })

    if cfg_save_raw_tensors_holder.get("save", False):
        torch.save(
            {
                "hidden_states": [h.cpu() for h in layer_hidden_states],
                "attentions": [a.cpu() for a in layer_attns_all],
                "prompt_len": prompt_len,
                "num_vision_patches": num_vision_patches,
                "generated_token_ids": generated_token_ids,
            },
            cfg_save_raw_tensors_holder["path_fn"](0),
        )

    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action, per_token_diag


# Ugly but scoped: lets get_action_with_diagnostics optionally dump raw tensors without threading
# a dozen extra params through every call site in this first draft.
cfg_save_raw_tensors_holder = {"save": False, "path_fn": None}


@draccus.wrap()
def probe(cfg: ProbeConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must be set!"
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.unnorm_key or cfg.task_suite_name

    vla = get_model(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, f"unnorm_key {cfg.unnorm_key} not in vla.norm_stats"
    processor = get_processor(cfg)

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_key = f"{cfg.task_suite_name}--t{cfg.task_id}"
    if cfg.run_id_note:
        run_key += f"--{cfg.run_id_note}"
    out_path = os.path.join(cfg.local_log_dir, f"{run_key}--{DATE_TIME}.jsonl")
    out_file = open(out_path, "a")
    print(f"Writing per-step diagnostics to {out_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    init_states = task_suite.get_task_init_states(cfg.task_id)

    conditions = cfg.conditions.split(",")

    for condition in conditions:
        assert condition in CONDITIONS, f"Unknown condition '{condition}'. Available: {sorted(CONDITIONS)}"
        instruction_map = CONDITIONS[condition]
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        if instruction_map is not None:
            assert task.name in instruction_map, f"No '{condition}' instruction for task '{task.name}'"
            task_description = instruction_map[task.name]
        print(f"\n=== condition={condition!r}  instruction={task_description!r} ===")

        # [2026-09-06] Bowl-distance instrumentation, borrowed from probe_bowl_attraction.py, so this
        # run's own per-episode "did it approach the target/distractor/neither" label comes from the
        # SAME rollout as the mechanistic diagnostics -- a separately-launched probe_bowl_attraction.py
        # run was tried first and found to diverge from this script's own rollouts after a handful of
        # steps (GPU floating-point non-determinism across process launches, same root cause as the
        # sdpa/eager finding above), so per-episode joins across two separate runs are NOT valid; only
        # instrumenting both signals in one rollout is.
        base = env.env
        bowl_names = sorted(n for n in base.obj_body_id if n.startswith("akita_black_bowl"))
        bowl_body_ids = {n: base.obj_body_id[n] for n in bowl_names}
        near_thresh_m = 0.08

        # num_vision_patches is prompt/image-shape invariant for this checkpoint -- compute once.
        env.reset()
        obs0 = env.set_init_state(init_states[0])
        img0 = get_libero_image(obs0, 224)
        prompt0 = _build_prompt(cfg.pretrained_checkpoint, task_description)
        inputs0 = processor(prompt0, Image.fromarray(img0).convert("RGB")).to(DEVICE, dtype=torch.bfloat16)
        num_vision_patches = _get_num_vision_patches(vla, inputs0)
        print(f"  num_vision_patches={num_vision_patches}")

        for episode_idx in range(cfg.num_trials):
            env.reset()
            obs = env.set_init_state(init_states[episode_idx])

            if cfg.save_raw_tensors:
                raw_dir = os.path.join(cfg.local_log_dir, "raw", f"{run_key}--{condition}--ep{episode_idx}")
                os.makedirs(raw_dir, exist_ok=True)
                cfg_save_raw_tensors_holder["save"] = True
                cfg_save_raw_tensors_holder["path_fn"] = lambda step, d=raw_dir: os.path.join(d, f"envstep_UNSET--tok{step}.pt")

            t = 0
            instrumented_steps = 0
            max_steps = cfg.max_env_steps_to_instrument
            done = False
            episode_records = []  # buffered so the final `success` label can be attached to every
                                   # step's record once known -- it isn't known until the episode ends
            min_dist = {n: float("inf") for n in bowl_names}
            first_near_step = {n: None for n in bowl_names}
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                img = get_libero_image(obs, 224)
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Same distance bookkeeping as probe_bowl_attraction.py, computed in THIS rollout.
                eef_pos = np.array(obs["robot0_eef_pos"])
                dist_by_bowl = {}
                for n, bid in bowl_body_ids.items():
                    bowl_pos = np.array(base.sim.data.body_xpos[bid])
                    d = float(np.linalg.norm(eef_pos - bowl_pos))
                    dist_by_bowl[n] = d
                    if d < min_dist[n]:
                        min_dist[n] = d
                    if d < near_thresh_m and first_near_step[n] is None:
                        first_near_step[n] = t
                if cfg.save_raw_tensors:
                    cfg_save_raw_tensors_holder["path_fn"] = (
                        lambda step, d=raw_dir, envstep=t: os.path.join(d, f"envstep{envstep}--tok{step}.pt")
                    )

                action, per_token_diag = get_action_with_diagnostics(
                    vla, processor, cfg.pretrained_checkpoint, observation, task_description,
                    cfg.unnorm_key, num_vision_patches, center_crop=cfg.center_crop,
                )

                rec = {
                    "task_id": cfg.task_id,
                    "task_name": task.name,
                    "condition": condition,
                    "episode_idx": episode_idx,
                    "env_step": t,
                    "dist_by_bowl": dist_by_bowl,
                    "per_token_diag": per_token_diag,
                }
                episode_records.append(rec)

                obs, reward, done, info = env.step(action.tolist())
                instrumented_steps += 1
                t += 1
                if done or instrumented_steps >= max_steps:
                    break

            # Same "first bowl actually reached for" derivation as probe_bowl_attraction.py.
            reached = [(step, n) for n, step in first_near_step.items() if step is not None]
            reached.sort()
            first_bowl = reached[0][1] if reached else None
            first_bowl_is_target = (first_bowl == "akita_black_bowl_1") if first_bowl else None

            for rec in episode_records:
                rec["success"] = bool(done)
                rec["instrumented_steps"] = instrumented_steps
                rec["min_dist_by_bowl"] = min_dist
                rec["first_near_step_by_bowl"] = first_near_step
                rec["first_bowl_approached"] = first_bowl
                rec["first_bowl_is_target"] = first_bowl_is_target
                out_file.write(json.dumps(rec) + "\n")
            out_file.flush()

            print(
                f"  ep {episode_idx}: success={bool(done)} instrumented_steps={instrumented_steps} "
                f"first_bowl={first_bowl}"
            )

        env.close()

    out_file.close()
    print(f"\nDone. Records at {out_path}")


if __name__ == "__main__":
    probe()
