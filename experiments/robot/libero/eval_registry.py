"""
eval_registry.py

Single canonical registry mapping a benchmark "split" (as specified in
vla_ws/docs/benchmark_split_plan.md) to the concrete (task_suite_name, unnorm_key,
condition) triple `run_libero_eval.py` needs, plus the instruction dict each
`condition` resolves to. This is the one place that knows how splits map onto
LIBERO task suites -- docker/openvla_libero/run_eval.sh and any future caller
should only ever pass `--split`, never hand-set suite/condition themselves.

Adding a split = add one entry to SPLITS (see vla_ws/CLAUDE.md "How to add a
benchmark split").
"""

from experiments.robot.libero.instructions import (
    LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS,
    LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS,
    LIBERO_SPATIAL_LENGTH_CONTROL_INFIX_INSTRUCTIONS,
    LIBERO_SPATIAL_LENGTH_CONTROL_SUFFIX_INSTRUCTIONS,
    LIBERO_SPATIAL_PARAPHRASE_LEXICAL_INSTRUCTIONS,
    LIBERO_SPATIAL_PARAPHRASE_SYNTACTIC_INSTRUCTIONS,
    LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_ADJACENT_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_BESIDE_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NEAR_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NOVEL_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_REGION_V2_INSTRUCTIONS,
    LIBERO_SPATIAL_TARGET_CUE_REGION_V3_INSTRUCTIONS,
)

# condition -> instruction dict (None = use LIBERO's own default task language)
CONDITIONS = {
    "default": None,
    "negative_contrast": LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS,
    "positive_contrast": LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS,
    "hardneg": LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS,
    "target_cue_region": LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS,
    "target_cue_landmark": LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS,
    "target_cue_proximity_novel": LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NOVEL_INSTRUCTIONS,
    "length_control_infix": LIBERO_SPATIAL_LENGTH_CONTROL_INFIX_INSTRUCTIONS,
    "length_control_suffix": LIBERO_SPATIAL_LENGTH_CONTROL_SUFFIX_INSTRUCTIONS,
    "paraphrase_lexical": LIBERO_SPATIAL_PARAPHRASE_LEXICAL_INSTRUCTIONS,
    "paraphrase_syntactic": LIBERO_SPATIAL_PARAPHRASE_SYNTACTIC_INSTRUCTIONS,
    "target_cue_region_v2": LIBERO_SPATIAL_TARGET_CUE_REGION_V2_INSTRUCTIONS,
    "target_cue_region_v3": LIBERO_SPATIAL_TARGET_CUE_REGION_V3_INSTRUCTIONS,
    "target_cue_proximity_beside": LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_BESIDE_INSTRUCTIONS,
    "target_cue_proximity_near": LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NEAR_INSTRUCTIONS,
    "target_cue_proximity_adjacent": LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_ADJACENT_INSTRUCTIONS,
}

# split_id -> (task_suite_name, unnorm_key, condition, description)
# unnorm_key is always "libero_spatial" for these variants because they all
# fine-tune-evaluate the same libero_spatial_no_noops checkpoint; only the
# scene/prompt changes.
SPLITS = {
    # --- Split 1: Prompt Sensitivity Probe (2-bowl baseline scene) ---
    "spatial/default": (
        "libero_spatial", "libero_spatial", "default",
        "Baseline: 2 bowls, default (target-only) prompt.",
    ),
    "spatial/negative_contrast": (
        "libero_spatial", "libero_spatial", "negative_contrast",
        "2 bowls, prompt names AND negates the distractor.",
    ),
    "spatial/positive_contrast": (
        "libero_spatial", "libero_spatial", "positive_contrast",
        "2 bowls, prompt mentions the distractor location without negating it.",
    ),
    # Length controls for the two contrast conditions above: native wording kept
    # verbatim, a content-free politeness clause added at the contrast clause's
    # position -- separates "longer prompt" from "second referent / off-template".
    "spatial/length_control_infix": (
        "libero_spatial", "libero_spatial", "length_control_infix",
        "2 bowls, native prompt + ', if it is not too much trouble for you,' where negative_contrast's "
        "'not the one ...' clause sits.",
    ),
    "spatial/length_control_suffix": (
        "libero_spatial", "libero_spatial", "length_control_suffix",
        "2 bowls, native prompt + '; thank you so very much in advance for your help with this' where "
        "positive_contrast's '; the other black bowl is ...' clause sits.",
    ),

    # --- Split 2: Distractor Placement Probe (3-bowl scenes, default prompt) ---
    "spatial_3bowl/irrelevant": (
        "libero_spatial_3bowl_front", "libero_spatial", "default",
        "3rd bowl always at main_table_table_front (literal front edge of the table, far from "
        "every object in every task), except 4 tasks (next_to_the_ramekin, on_the_cookie_box, "
        "on_the_ramekin, next_to_the_cookie_box) that fall back to table_center because their "
        "own bowls sit within ~0.10-0.13m of table_front. 2026-08-27 fine-tune: table_front "
        "+0.05m further front, table_center fallback +0.05m further back (still ~0.29m clear "
        "of stove_region). Second redefinition -- see irrelevant_v1_legacy and "
        "center_fixed_legacy below for the prior two.",
    ),
    "spatial_3bowl/irrelevant_v1_legacy": (
        "libero_spatial_3bowl_neutral", "libero_spatial", "default",
        "First redefinition of 'irrelevant': 3rd bowl at a per-task region chosen to be off "
        "the target-to-plate reach path and distance-matched to semantic/landmark, rather "
        "than one shared coordinate -- but reused next_to_ramekin_region (itself another "
        "task's real target landmark) for 5/10 tasks. Kept only so benchmark_split_result.md's "
        "existing 88.8% number stays attributable -- do not treat as the current 'irrelevant' "
        "condition, see benchmark_split_plan.md Split 2's redefinition note.",
    ),
    "spatial_3bowl/center_fixed_legacy": (
        "libero_spatial_3bowl", "libero_spatial", "default",
        "Retired definition of 'irrelevant': 3rd bowl always at table_center/table_front "
        "regardless of task. Kept only so benchmark_split_result.md's existing 80.2% number (Exp 2) "
        "stays attributable -- do not treat as the current 'irrelevant' condition, see "
        "benchmark_split_plan.md Split 2's confound note.",
    ),
    "spatial_3bowl/semantic": (
        "libero_spatial_3bowl_semantic2", "libero_spatial", "default",
        "3rd bowl at a named landmark different from the target's own landmark. Second "
        "redefinition -- only task 4 (in the top drawer) changed vs. semantic_v1_legacy, "
        "moved from next_to_plate_region (~0.58m from the target, outside the semantic band "
        "and effectively behaving like 'irrelevant') to between_plate_ramekin_region (~0.48m).",
    ),
    "spatial_3bowl/semantic_v1_legacy": (
        "libero_spatial_3bowl_semantic", "libero_spatial", "default",
        "First definition of 'semantic': identical to the current one except task 4's bowl_3 "
        "sat at next_to_plate_region (~0.58m from the target, outside the 0.33-0.50m band the "
        "other 9 tasks land in). Kept only so benchmark_split_result.md's existing 84.8% "
        "number stays attributable -- do not treat as the current 'semantic' condition.",
    ),
    "spatial_3bowl/landmark": (
        "libero_spatial_3bowl_hardneg", "libero_spatial", "default",
        "3rd bowl near the target's OWN landmark but farther away (hard negative).",
    ),
    # "path" (3rd bowl between target and plate) is specified but not yet
    # authored -- see benchmark_split_plan.md Split 2, open design questions.

    # --- Split 3: Scene Complexity Probe ---
    "spatial_3bowl/drawer_open": (
        "libero_spatial_3bowl_open", "libero_spatial", "default",
        "3 bowls + wooden cabinet's top drawer open, default prompt.",
    ),

    # --- Split 2 x Split 1: hard-negative scene with a disambiguating prompt ---
    "spatial_3bowl/landmark_with_hardneg_prompt": (
        "libero_spatial_3bowl_hardneg", "libero_spatial", "hardneg",
        "3-bowl hard-negative scene, prompt disambiguates target vs. the near hard negative.",
    ),

    # --- Split 4: Surface vs Landmark Grounding Probe, gap-fill cells ---
    # 4 of 6 (target-family, distractor-family) cells already exist inside the
    # libero_spatial baseline -- see GROUNDING_PROBE_CELLS below, no separate
    # split entry needed (use `spatial/default` filtered to those task ids via
    # --task_ids). These two cells needed a new scene (existing distractor bowl
    # moved to a different pre-existing region; canonical libero_spatial untouched).
    "grounding/surface_landmark": (
        "libero_spatial_grounding_surface_landmark", "libero_spatial", "default",
        "Surface-cue target (on the ramekin) + landmark-cue distractor (next to cookie box).",
    ),
    "grounding/region_surface": (
        "libero_spatial_grounding_region_surface", "libero_spatial", "default",
        "Region-cue target (table center) + surface-cue distractor (on the stove).",
    ),

    # --- Split 4b: Target Cue-Type Probe (same libero_spatial scene as spatial/default;
    # only the TARGET's phrasing changes, distractor never mentioned -- see
    # benchmark_split_plan.md Split 4's 4b section). Each condition's instruction dict
    # only covers the task ids listed below -- run_libero_eval.py asserts
    # task.name in instruction_map, so these MUST be run with --task_ids
    # restricted to that subset, never the full 10-task suite. ---
    "grounding/target_cue_region": (
        "libero_spatial", "libero_spatial", "target_cue_region",
        "Target rephrased as a table-zone/region description instead of its native "
        "landmark or surface cue. Run with --task_ids 0 1 3 5 6 7 8 9 (task 2 is "
        "already native region, task 4 is containment -- both excluded).",
    ),
    "grounding/target_cue_landmark": (
        "libero_spatial", "libero_spatial", "target_cue_landmark",
        "Surface-family target rephrased as a landmark ('next to X') cue instead of "
        "its native surface ('on X') cue -- disclosed-approximate, see "
        "benchmark_split_plan.md's truthfulness-tier table. Run with --task_ids 3 5 7 9 "
        "(the only tasks where this rephrasing is defined).",
    ),

    # --- Split 4c: Familiar vs. Novel Proximity-Cue Probe (same 4 surface-family
    # tasks and scene as target_cue_landmark; isolates whether that condition's
    # drop tracks relation-type change or exact-template familiarity -- see
    # benchmark_split_plan.md Split 4's 4c section). ---
    "grounding/target_cue_proximity_novel": (
        "libero_spatial", "libero_spatial", "target_cue_proximity_novel",
        "Surface-family target rephrased with a proximity synonym ('close to X') "
        "never used in any native libero_spatial prompt -- same disclosed-approximate "
        "reading as target_cue_landmark ('next to X'), but lexically novel rather than "
        "a familiar template borrowed from tasks 0/1/6/8. Run with --task_ids 3 5 7 9.",
    ),
    "grounding/target_cue_proximity_beside": (
        "libero_spatial", "libero_spatial", "target_cue_proximity_beside",
        "As target_cue_proximity_novel, with 'beside X'. Run with --task_ids 3 5 7 9.",
    ),
    "grounding/target_cue_proximity_near": (
        "libero_spatial", "libero_spatial", "target_cue_proximity_near",
        "As target_cue_proximity_novel, with 'near X'. Run with --task_ids 3 5 7 9.",
    ),
    "grounding/target_cue_proximity_adjacent": (
        "libero_spatial", "libero_spatial", "target_cue_proximity_adjacent",
        "As target_cue_proximity_novel, with 'adjacent to X'. Run with --task_ids 3 5 7 9.",
    ),

    # --- Split 4b paraphrase controls: region-cue rewordings (same zones as
    # target_cue_region) and the same-cue diagonal (native cue type, truthful,
    # reworded) -- see benchmark_split_plan.md Split 4's 4b paraphrase section. ---
    "grounding/target_cue_region_v2": (
        "libero_spatial", "libero_spatial", "target_cue_region_v2",
        "As target_cue_region, reworded as '<zone> area/part of the table'. "
        "Run with --task_ids 0 1 3 5 6 7 8 9.",
    ),
    "grounding/target_cue_region_v3": (
        "libero_spatial", "libero_spatial", "target_cue_region_v3",
        "As target_cue_region, reworded side-first ('on the left side of the table, "
        "toward the back'). Run with --task_ids 0 1 3 5 6 7 8 9.",
    ),
    "grounding/paraphrase_lexical": (
        "libero_spatial", "libero_spatial", "paraphrase_lexical",
        "Target in its native cue type, exactly truthful, one synonym swap or argument "
        "reorder ('beside X', 'on top of X', ...). All 10 tasks.",
    ),
    "grounding/paraphrase_syntactic": (
        "libero_spatial", "libero_spatial", "paraphrase_syntactic",
        "Target in its native words, recast as a relative clause ('the black bowl that "
        "is <where>'). All 10 tasks.",
    ),
}

# --- Split 4 metadata: relation family of the TARGET in each libero_spatial task ---
TARGET_RELATION_FAMILY = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "landmark",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": "landmark",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate": "region",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": "surface",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": "containment",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "surface",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": "landmark",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "surface",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": "landmark",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": "surface",
}

# --- Split 4 metadata: relation family of the existing 2nd-bowl DISTRACTOR in
# each libero_spatial task (used to find which (target,distractor) family
# pairs already exist in the baseline scene, no new BDDL needed) ---
DISTRACTOR_RELATION_FAMILY = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "landmark",  # next_to_ramekin_region
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": "landmark",  # next_to_box_region
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate": "landmark",  # next_to_plate_region
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": "surface",  # wooden_cabinet top
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": "surface",  # wooden_cabinet top
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "surface",  # cookies_1
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": "surface",  # stove cook_region
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "surface",  # wooden_cabinet top
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": "landmark",  # next_to_ramekin_region
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": "surface",  # stove cook_region
}

# (target_family, distractor_family) -> list of (task_suite_name, task_name).
# Built from the tables above plus the two gap-fill suites. "containment"
# target rows are excluded -- Split 4's matrix only covers landmark/surface/region.
GROUNDING_PROBE_CELLS = {}
for _task, _tfam in TARGET_RELATION_FAMILY.items():
    if _tfam == "containment":
        continue
    _dfam = DISTRACTOR_RELATION_FAMILY[_task]
    GROUNDING_PROBE_CELLS.setdefault((_tfam, _dfam), []).append(("libero_spatial", _task))
GROUNDING_PROBE_CELLS[("surface", "landmark")] = [
    ("libero_spatial_grounding_surface_landmark", "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate")
]
GROUNDING_PROBE_CELLS[("region", "surface")] = [
    ("libero_spatial_grounding_region_surface", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
]
del _task, _tfam, _dfam


def resolve_split(split_id):
    """Look up (task_suite_name, unnorm_key, condition) for a split id."""
    if split_id not in SPLITS:
        raise KeyError(f"Unknown split '{split_id}'. Available: {sorted(SPLITS)}")
    suite, unnorm_key, condition, _desc = SPLITS[split_id]
    return suite, unnorm_key, condition
