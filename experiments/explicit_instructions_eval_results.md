# OpenVLA · LIBERO-Spatial — Explicit vs. Default Instruction Evaluation

**Model:** `openvla-7b` LoRA (r32) fine-tuned on `libero_spatial_no_noops` (bf16 + FlashAttention-2, center-crop)
**Suite:** `libero_spatial` — 10 tasks × 50 trials = **500 rollouts** per condition
**Env:** `openvla-libero:cuda12.1` Docker (mujoco 2.3.2, robosuite 1.4.1, EGL headless on NVIDIA)
**Hardware:** 5× H200 (GPUs 0–4), tasks round-robin sharded 2/GPU
**Seed:** 7

The only difference between the two conditions is the **prompt string**. Each scene contains two
identical black bowls (target + distractor); the scene, objects, and initial states are identical.

- **Default:** names only the target — e.g. *"pick up the black bowl on the stove and place it on the plate."*
- **Explicit:** additionally names where the distractor is — *"…, not the one on top of the wooden cabinet, …"*

---

## Headline

| Condition | Overall success | Rollouts |
|---|---:|---:|
| **Default (ordinary)** | **84.0%** | 420 / 500 |
| **Explicit (distractor-aware)** | **36.8%** | 184 / 500 |
| **Δ (explicit − default)** | **−47.2 pts** | |

Adding the explicit distractor clause **hurts** the baseline model across almost every task.

---

## Per-task comparison (by task ID)

| ID | Default | Explicit | Δ | Target |
|---:|--:|--:|--:|---|
| 5 | 94% | 4% | **−90** | on the ramekin |
| 0 | 92% | 94% | +2 | between the plate and the ramekin |
| 2 | 92% | 2% | **−90** | table center |
| 6 | 90% | 72% | −18 | next to the cookie box |
| 1 | 84% | 32% | −52 | next to the ramekin |
| 3 | 84% | 52% | −32 | on the cookie box |
| 8 | 84% | 36% | −48 | next to the plate |
| 4 | 76% | 64% | −12 | in the top drawer of the wooden cabinet |
| 7 | 72% | 4% | −68 | on the stove |
| 9 | 72% | 8% | −64 | on the wooden cabinet |

---

## Observations

- **Baseline is strong and uniform with ordinary instructions** — 72–94% across all 10 tasks.
- **The distractor clause regresses performance almost everywhere (−47 pts overall).** Rather than
  aiding disambiguation, the extra *"…not the one on the X…"* phrasing appears to confuse the policy.
- **Largest drops** occur where the distractor phrase names a salient surface: ramekin (−90),
  table center (−90), stove (−68), wooden cabinet (−64).
- **Only unaffected task:** "between the plate and the ramekin" (92% → 94%), the one already phrased relationally.

---

## Explicit instructions used (distractor clause in **bold**)

| Default target | Explicit instruction |
|---|---|
| between the plate and the ramekin | pick up the black bowl between the plate and the ramekin, **not the one next to the ramekin**, and place it on the plate |
| from table center | pick up the black bowl at the center of the table, **not the one next to the plate**, and place it on the plate |
| in the top drawer of the cabinet | pick up the black bowl inside the top drawer of the wooden cabinet, **not the one on top of the cabinet**, and place it on the plate |
| next to the cookie box | pick up the black bowl next to the cookie box, **not the one on the stove**, and place it on the plate |
| next to the plate | pick up the black bowl next to the plate, **not the one next to the ramekin**, and place it on the plate |
| next to the ramekin | pick up the black bowl next to the ramekin, **not the one next to the cookie box**, and place it on the plate |
| on the cookie box | pick up the black bowl on top of the cookie box, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the ramekin | pick up the black bowl on top of the ramekin, **not the one on top of the cookie box**, and place it on the plate |
| on the stove | pick up the black bowl on the stove, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the wooden cabinet | pick up the black bowl on top of the wooden cabinet, **not the one on the stove**, and place it on the plate |

---

## Reproduce

```bash
cd /home/ec2-user/aliy/vla_ws/docker/openvla_libero

# Default (ordinary) instructions, 5 GPUs:
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh

# Explicit distractor-aware instructions, 5 GPUs:
USE_EXPLICIT_PROMPT=True  GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh
```

**Config**
- Checkpoint: `/home/ec2-user/wenhan/openvla/checkpoints/baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug`
- Scenes / init states / BDDLs: `vla_ws/LIBERO/libero/libero/bddl_files/libero_spatial/` (two-bowl variants)
- Action un-norm stats: checkpoint `dataset_statistics.json` (key `libero_spatial_no_noops`)
- Note: the `modified_libero_rlds` RLDS path is training data and is **not** read at eval time.

**Artifacts**
- Per-shard logs: `experiments/logs/EVAL-…--{default,explicit}_instructions--shard{0..4}of5.txt`
- Rollout videos: `experiments/rollouts/<date>/` (tagged with task + success)
