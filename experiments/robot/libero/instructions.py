"""
instructions.py

Instruction (prompt condition) sets for the LIBERO Spatial suite and its scene
variants. Every dict here is keyed by `task.name` (the BDDL filename without
extension) and phrased as a single imperative clause so it slots directly into
the OpenVLA prompt template
    "In: What action should the robot take to {instruction.lower()}?\nOut:"
without any template changes.

Each scene contains TWO identical black bowls: the target (`akita_black_bowl_1`)
and a distractor (`akita_black_bowl_2`). The default LIBERO instructions only
describe the target ("pick up the black bowl <where> ..."). The sets below are
`benchmark_split.md` Split 1's three prompt conditions ("Prompt Sensitivity
Probe") over that same 2-bowl scene, plus the hard-negative condition used by
`libero_spatial_3bowl_hardneg`. See eval_registry.py for how a `--condition`
CLI flag selects one of these at eval time.
"""

# ---------------------------------------------------------------------------
# Split 1 "Negative contrast": names AND negates the distractor
# ("..., not the one <where>, ..."). Verified against each task's actual
# `akita_black_bowl_2` init region in the corresponding BDDL file.
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl between the plate and the ramekin, not the one next to the ramekin, and place it on the plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl at the center of the table, not the one next to the plate, and place it on the plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl inside the top drawer of the wooden cabinet, not the one on top of the cabinet, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl next to the cookie box, not the one on the stove, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl next to the plate, not the one next to the ramekin, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl next to the ramekin, not the one next to the cookie box, and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on top of the cookie box, not the one on top of the wooden cabinet, and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on top of the ramekin, not the one on top of the cookie box, and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on the stove, not the one on top of the wooden cabinet, and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl on top of the wooden cabinet, not the one on the stove, and place it on the plate",
}


# ---------------------------------------------------------------------------
# Split 1 "Positive contrast": names the distractor's location WITHOUT negating
# it (no "not the one ..." clause) -- tests whether merely mentioning a second
# bowl hurts, independent of the negation/contrast itself. Same distractor
# locations as LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS above, just phrased as a
# plain mention. See benchmark_split.md Split 1 for the derived metrics
# (Distractor Mention Drop, Negation-specific Drop).
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl between the plate and the ramekin and place it on the plate; the other black bowl is next to the ramekin",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl at the center of the table and place it on the plate; the other black bowl is next to the plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl inside the top drawer of the wooden cabinet and place it on the plate; the other black bowl is on top of the cabinet",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl next to the cookie box and place it on the plate; the other black bowl is on the stove",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl next to the plate and place it on the plate; the other black bowl is next to the ramekin",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl next to the ramekin and place it on the plate; the other black bowl is next to the cookie box",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on top of the cookie box and place it on the plate; the other black bowl is on top of the wooden cabinet",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on top of the ramekin and place it on the plate; the other black bowl is on top of the cookie box",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on the stove and place it on the plate; the other black bowl is on top of the wooden cabinet",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl on top of the wooden cabinet and place it on the plate; the other black bowl is on the stove",
}


# ---------------------------------------------------------------------------
# Split 4b "Target Cue-Type Probe": rephrases ONLY the target description using
# an alternate relation-family cue (landmark "next to X" / surface "on X" /
# region "table-zone" wording), distractor never mentioned, scene/init states
# identical to `libero_spatial` (`spatial/default`). Isolates cue TYPE from
# both scene content (Split 4a's axis) and distractor mention (Split 1's axis
# -- see `benchmark_split.md` Split 4's 4b section for why that axis is
# dropped here). Only covers tasks where the alternate phrasing is a truthful
# or disclosed-approximate description of that task's actual bowl placement;
# tasks not covered are intentionally absent, not defaulted -- run this
# condition with `--task_ids` restricted to this dict's keys (see
# eval_registry.SPLITS descriptions for `grounding/target_cue_region` and
# `grounding/target_cue_landmark`), since run_libero_eval.py asserts
# `task.name in instruction_map` and does not silently skip missing tasks.
# ---------------------------------------------------------------------------

# Landmark-family tasks (0, 1, 6, 8) already use this cue type natively via
# `spatial/default` and are intentionally omitted -- rephrasing them here
# would just duplicate that baseline. Task 2 (region, table_center) has no
# nameable landmark object nearby and task 4 (containment) is out of scope
# for Split 4's matrix (see benchmark_split.md).
LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl at the back of the table, just left of center, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl at the far back-left of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl near the center of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl at the back-left of the table and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl at the front-right of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl at the front-left of the table and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl at the far back of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl at the front of the table, just right of center, and place it on the plate",
}


# Surface-family tasks only (3, 5, 7, 9): the bowl rests ON the named object,
# so "next to X" is a disclosed-approximate (co-located, not exact) reading --
# see benchmark_split.md Split 4's truthfulness-tier table. Landmark-family
# tasks already use this cue natively (omitted, see above); region-family
# task 2 has no nameable object near `table_center` to reference.
LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS = {
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl next to the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl next to the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl next to the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl next to the wooden cabinet and place it on the plate",
}


# ---------------------------------------------------------------------------
# Hard-negative distractor instructions (for the `libero_spatial_3bowl_hardneg`
# suite). There the third bowl (`akita_black_bowl_3`) sits near the SAME landmark
# as the target but farther from it, so the plain instruction is genuinely
# ambiguous. These explicit versions name the target by its relative distance /
# position vs. that hard negative ("the closest one", "not the one in front of…").
# Keyed by task.name, same as above.
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl directly between the plate and the ramekin, not the one in front of them, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl closest to the ramekin, not the one farther from it, and place it on the plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl at the center of the table, not the one off to the side, and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on top of the cookie box, not the one on the table next to it, and place it on the plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl inside the top drawer of the wooden cabinet, not the one on the table in front of the cabinet, and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on top of the ramekin, not the one on the table next to it, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl closest to the cookie box, not the one farther from it, and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on the stove, not the one on the table in front of it, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl closest to the plate, not the one farther from it, and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl on top of the wooden cabinet, not the one on the table in front of it, and place it on the plate",
}
