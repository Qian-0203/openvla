# OpenVLA · LIBERO-Spatial — Instruction & Scene-Robustness Evaluations

How a `libero_spatial`-fine-tuned OpenVLA baseline holds up as we change **the prompt** and
**the scene** — while keeping the model, checkpoint, and seed fixed. Each experiment changes
exactly one thing and is reported against the same 84.0% baseline.

## At a glance

| # | Experiment | What changed | Overall success | vs. baseline |
|--:|---|---|--:|--:|
| — | **Baseline** — default prompt, 2 bowls | — | **84.0%** | — |
| 1 | Explicit prompt | prompt names the distractor too | 36.8% | **−47.2** |
| 2 | Extra distractor | +1 bowl (3 total), default prompt | 80.2% | −3.8 |
| 3 | Cluttered scene | 3 bowls **+ top drawer open**, default prompt | 60.0% (raw) / **73.1%**† | −24.0 (raw) / **−11.7**† |

† Adjusted: tasks 3, 6, 7 excluded — rollout review showed the open drawer **physically blocks their
trajectories**, so those failures measure scene feasibility, not policy robustness (see Exp 3). The
adjusted Δ compares against the baseline recomputed on the same 7 tasks (84.9%).

**Headlines.** (1) Naming the distractor in the prompt *badly* hurts the policy (−47 pts). (2) Adding
a second distractor bowl barely dents overall success (−3.8 pts) but sinks one specific task. (3)
Opening the cabinet drawer looks like a broad −20 pt collapse at first glance, but roughly half of
that is an environment artifact: on tasks 3, 6, and 7 the protruding drawer blocks the motion path
outright. Excluding those, the drawer's *policy-attributable* cost is **−11.7 pts** (73.1% on the
7 feasible tasks) — still ~3× the extra bowl's damage, but a robustness gap, not a collapse.

---

## Setup (common to all experiments)

- **Model:** `openvla-7b` LoRA (r32) fine-tuned on `libero_spatial_no_noops` (bf16 + FlashAttention-2, center-crop).
- **Base suite:** `libero_spatial` — 10 tasks; each = *"pick up the black bowl \<location\> and place it on the plate."*
- **Protocol:** 50 trials/task = **500 rollouts** per condition. Seed 7. Success = LIBERO goal met.
- **Env:** `openvla-libero:cuda12.1` Docker (mujoco 2.3.2, robosuite 1.4.1, EGL headless).
- **Hardware:** 5× H200 (GPUs 0–4), tasks round-robin sharded 2/GPU.
- **Task ids** are identical across every suite below, so columns line up 1:1:

  | id | target location | id | target location |
  |--:|---|--:|---|
  | 0 | between the plate and the ramekin | 5 | on the ramekin |
  | 1 | next to the ramekin | 6 | next to the cookie box |
  | 2 | table center | 7 | on the stove |
  | 3 | on the cookie box | 8 | next to the plate |
  | 4 | in the top drawer of the wooden cabinet | 9 | on the wooden cabinet |

- **Noise:** at 50 trials/task, 1 standard error ≈ ±5–7 pts, so treat single-task swings ≲10 pts as noise.

---

## Experiment 1 — Prompt phrasing: default vs. explicit (2 bowls)

**Question.** The scene has two identical black bowls (target + 1 distractor). Does *telling the model
which bowl to avoid* help? Only the **prompt string** changes; scene and init states are identical.

- **Default:** names only the target — *"pick up the black bowl on the stove and place it on the plate."*
- **Explicit:** also names the distractor — *"…, not the one on top of the wooden cabinet, …"*

| Condition | Overall | Rollouts |
|---|--:|--:|
| **Default (ordinary)** | **84.0%** | 420 / 500 |
| **Explicit (distractor-aware)** | **36.8%** | 184 / 500 |
| **Δ** | **−47.2 pts** | |

| id | target | distractor named in prompt | Default | Explicit | Δ |
|--:|---|---|--:|--:|--:|
| 0 | between the plate and the ramekin | next to the ramekin | 92% | 94% | +2 |
| 1 | next to the ramekin | next to the cookie box | 84% | 32% | −52 |
| 2 | table center | next to the plate | 92% | 2% | **−90** |
| 3 | on the cookie box | on top of the wooden cabinet | 84% | 52% | −32 |
| 4 | in the top drawer | on top of the cabinet | 76% | 64% | −12 |
| 5 | on the ramekin | on top of the cookie box | 94% | 4% | **−90** |
| 6 | next to the cookie box | on the stove | 90% | 72% | −18 |
| 7 | on the stove | on top of the wooden cabinet | 72% | 4% | −68 |
| 8 | next to the plate | next to the ramekin | 84% | 36% | −48 |
| 9 | on the wooden cabinet | on the stove | 72% | 8% | −64 |

**Takeaway.** The explicit "…not the one on X…" clause *confuses* the policy instead of disambiguating
it (−47 pts overall). Damage is worst where the clause names a salient surface (ramekin, table center,
stove, cabinet). The only unhurt task (0) was already phrased relationally.

<details><summary>Exact explicit prompts used (distractor clause in <b>bold</b>)</summary>

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

</details>

---

## Experiment 2 — Extra distractor: 2 bowls vs. 3 bowls (default prompt)

**Question.** Keep the ordinary (target-only) prompt but make the scene harder: add a **third**
`akita_black_bowl` (a second distractor) at the open table center — table front for task 2, whose
target already sits at center. Does more visual ambiguity break the policy?

| Condition (default prompt) | Overall | Rollouts |
|---|--:|--:|
| **2 bowls** (target + 1 distractor) | **84.0%** | 420 / 500 |
| **3 bowls** (target + 2 distractors) | **80.2%** | 401 / 500 |
| **Δ** | **−3.8 pts** | |

| id | target | 2-bowl | 3-bowl | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 76% | −16 |
| 1 | next to the ramekin | 84% | 90% | +6 |
| 2 | table center | 92% | 94% | +2 |
| 3 | on the cookie box | 84% | 86% | +2 |
| 4 | in the top drawer | 76% | 84% | +8 |
| 5 | on the ramekin | 94% | 84% | −10 |
| 6 | next to the cookie box | 90% | 44% | **−46** |
| 7 | on the stove | 72% | 82% | +10 |
| 8 | next to the plate | 84% | 88% | +4 |
| 9 | on the wooden cabinet | 72% | 74% | +2 |

**Takeaway.** Overall success is **robust** to one extra distractor (−3.8 pts) — but the loss is
**concentrated**: task 6 (*next to the cookie box*) collapses 90%→44% because the added center bowl
lands between the cookie box and the plate, and it is next but little farther to the cookie box than the target.

**Init-state figure** — left = 2 bowls (`libero_spatial`), right = 3 bowls (`libero_spatial_3bowl`);
rows are task ids 0–9. Each panel is the real episode-0 state restored via `set_init_state`.

![2-bowl vs 3-bowl init states](figures/compare_2v3bowl_grid.png)

The extra free-body bowl enlarges the restored MuJoCo state vector (**92 → 105 dims**), so the 3-bowl
suite ships freshly generated `.pruned_init` files. Placement was checked across all 500 states: worst
bowl-to-bowl distance 0.122 m (bowl ⌀ ≈ 0.115 m) — no overlaps, valid heights.

---

## Experiment 3 — Cluttered scene: 3 bowls + open top drawer (default prompt)

**Question.** Keep the 3-bowl scene and add clutter/occlusion by **opening the wooden cabinet's top
("first") drawer** in every task. Does a protruding open drawer degrade the policy further?

### Blocked tasks — tasks 3, 6, 7 excluded from the adjusted overall

Rollout review showed that for **task 3** (*on the cookie box*), **task 6** (*next to the cookie
box*), and **task 7** (*on the stove*), the open top drawer **physically blocks the trajectory**: the
protruding drawer sits directly in the pick-and-place corridor these tasks must traverse, so the arm
cannot complete the motion regardless of what the policy predicts. Failures on these three tasks
therefore measure the *scene's physical feasibility*, not the *policy's robustness*, and they are
excluded from the adjusted overall below. (Task 6's 0% is the clearest case — a hard geometric block,
not a perception error.)

**Raw** (all 10 tasks) and **adjusted** (7 feasible tasks, 350 rollouts) overalls:

| Condition (default prompt) | Raw overall (10 tasks) | Adjusted overall (7 tasks†) |
|---|--:|--:|
| 3 bowls, drawer closed (Exp 2) | 80.2% (401 / 500) | 84.3% (295 / 350) |
| 3 bowls, **drawer open** | 60.0% (300 / 500) | **73.1%** (256 / 350) |
| **Δ (open − closed)** | −20.2 pts | **−11.1 pts** |

† Same-subset comparison: tasks {0, 1, 2, 4, 5, 8, 9} in both conditions. Against the original 2-bowl
baseline on the same 7 tasks (84.9%, 297/350), the adjusted drop is **−11.7 pts** — vs. the raw
headline of −24.0, which conflates policy degradation with physically infeasible tasks.

| id | target | 3-bowl | +drawer open | Δ | |
|--:|---|--:|--:|--:|---|
| 0 | between the plate and the ramekin | 76% | 66% | −10 | |
| 1 | next to the ramekin | 90% | 86% | −4 | |
| 2 | table center | 94% | 80% | −14 | |
| 3 | on the cookie box | 86% | 42% | −44 | ‡ blocked |
| 4 | in the top drawer | 84% | 84% | 0 | |
| 5 | on the ramekin | 84% | 84% | 0 | |
| 6 | next to the cookie box | 44% | 0% | −44 | ‡ blocked |
| 7 | on the stove | 82% | 46% | −36 | ‡ blocked |
| 8 | next to the plate | 88% | 68% | −20 | |
| 9 | on the wooden cabinet | 74% | 44% | −30 | |

‡ Trajectory physically obstructed by the open drawer — excluded from the adjusted overall.

**Takeaway.** The raw −20 pt drop overstates the policy effect: nearly half of it comes from the three
blocked tasks (3, 6, 7), where no policy could succeed because the open drawer occupies the motion
corridor. On the seven tasks that remain physically feasible, the open drawer still costs the policy
**−11.1 pts** vs. the closed-drawer scene — a genuine (occlusion/robustness) effect, concentrated
around the cabinet: task 9 (*on the wooden cabinet*, −30) and task 8 (*next to the plate*, −20) drop
well beyond noise, with smaller dips on 2 and 0. The two tasks that are *unaffected* (4 *in the top
drawer*, 5 *on the ramekin*) are telling: task 4's target is inside the drawer, so an open drawer is
expected there. The protruding drawer occludes the cabinet side of the table, which is exactly where
the still-degraded targets sit or are approached — but the effect is a robustness gap (~−11 pts), not
the collapse the raw number suggests.

**Init-state figure** — left = drawer closed (`libero_spatial_3bowl`), right = drawer open
(`libero_spatial_3bowl_open`); rows are task ids 0–9.

![3-bowl: drawer closed vs open](figures/compare_3bowl_closed_vs_open_grid.png)

Verified across all 500 open-drawer init states: bowls never overlap (worst separation 0.122 m) and the
cabinet top-drawer joint is open (qpos −0.141 m) in every trial. Adding the drawer state does not change
the state-vector size (105 dims) — the drawer joint already exists in the cabinet model; only its
position changes.

---

## Reproduce

All commands from `vla_ws/docker/openvla_libero/`. Each runs the same eval harness in Docker across
GPUs 0–4; `USE_EXPLICIT_PROMPT` toggles Exp 1's prompt.

```bash
# Exp 1 — baseline / explicit (2 bowls)
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh
USE_EXPLICIT_PROMPT=True  GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh

# Exp 2 — three bowls, default prompt
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_multigpu.sh

# Exp 3 — three bowls + open drawer, default prompt
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_open_multigpu.sh
```

**Config**
- Checkpoint: `/home/ec2-user/wenhan/openvla/checkpoints/baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug`
- Action un-norm stats: checkpoint key `libero_spatial_no_noops`. The variant suites pass
  `--unnorm_key libero_spatial` so their differing suite name still resolves to that key.
- The `modified_libero_rlds` RLDS path is training data and is **not** read at eval time.

**Scene variants** (new parallel suites; the canonical `libero_spatial` is untouched)

| Suite | Scene | BDDLs / init states |
|---|---|---|
| `libero_spatial` | 2 bowls (stock) | `LIBERO/libero/libero/{bddl_files,init_files}/libero_spatial/` |
| `libero_spatial_3bowl` | +1 bowl | `…/libero_spatial_3bowl/` |
| `libero_spatial_3bowl_open` | +1 bowl, top drawer open | `…/libero_spatial_3bowl_open/` |

Variant suites are registered in `LIBERO/libero/libero/benchmark/{__init__.py,libero_suite_task_map.py}`.
Init states (dim 105) were regenerated inside the eval Docker.

**Helper scripts** (`LIBERO/scripts/`)
- `gen_suite_init_states.py <suite>` — sample + save init states, render a contact sheet.
- `verify_3bowl_init_states.py` — assert bowls never overlap; report heights.
- `compare_2v3_bowl_init.py` — render the 2-bowl vs 3-bowl comparison figure.

**Artifacts**
- Per-shard logs: `experiments/logs/EVAL-<suite>-…--<condition>--shard{0..4}of5.txt`
- Rollout videos: `openvla/rollouts/<date>/` (tagged with task + success)
- Figures: `experiments/figures/`
