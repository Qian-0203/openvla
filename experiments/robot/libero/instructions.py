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
`benchmark_split_plan.md` Split 1's three prompt conditions ("Prompt Sensitivity
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
# plain mention. See benchmark_split_plan.md Split 1 for the derived metrics
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
# -- see `benchmark_split_plan.md` Split 4's 4b section for why that axis is
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
# for Split 4's matrix (see benchmark_split_plan.md).
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
# see benchmark_split_plan.md Split 4's truthfulness-tier table. Landmark-family
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
# Split 4c "Familiar vs. Novel Proximity-Cue Probe": same 4 surface-family
# tasks as LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS above, same
# disclosed-approximate "co-located with X" reading, but swaps "next to X"
# (the native landmark-family phrasing used verbatim in tasks 0/1/6/8) for
# "close to X" -- a proximity synonym that never appears in ANY of the 10
# native `libero_spatial` prompts. Isolates whether TARGET_CUE_LANDMARK's
# ~50pt drop (see benchmark_split_result.md Split 4's 4b write-up) comes from
# the relation-type change (surface -> proximity) or from matching a specific
# memorized sentence template, independent of relation type: if this condition
# drops about as much as TARGET_CUE_LANDMARK, the damage tracks relation type;
# if it drops substantially more, exact template familiarity is doing most of
# the work. Distractor never mentioned, scene/init states identical to
# `libero_spatial`. Run with `--task_ids 3 5 7 9` only (same constraint as
# TARGET_CUE_LANDMARK -- run_libero_eval.py asserts `task.name in
# instruction_map` and does not skip missing tasks).
# ---------------------------------------------------------------------------

LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NOVEL_INSTRUCTIONS = {
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl close to the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl close to the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl close to the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl close to the wooden cabinet and place it on the plate",
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


# ---------------------------------------------------------------------------
# Length-control probe (`benchmark_split_plan.md` Split 1's length control):
# Split 1's contrast conditions both lengthen the prompt AND deviate from the
# fine-tuning template, so their ~50pt drop can't yet be attributed to either
# alone (plan §11.3 gap 2). These keep every task's native `default` wording
# verbatim and add a content-free politeness clause -- no object, location, or
# second referent -- at the same position the contrast clause occupies:
#   - INFIX:  where negative_contrast's ", not the one <where>," sits.
#   - SUFFIX: where positive_contrast's "; the other black bowl is <where>" sits.
# A single fixed filler per condition (not per-task tuned) so the manipulation
# is identical across tasks. Fillers were picked to match the mean extra
# Llama-2 tokens (vs. native) of the clause they stand in for: infix +11
# (negative_contrast: mean +11.1, range 9-13), suffix +13 (positive_contrast:
# mean +13.1, range 11-15).
# All 10 tasks, scene/init states identical to `libero_spatial`.
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_LENGTH_CONTROL_INFIX_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl between the plate and the ramekin, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl from table center, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl in the top drawer of the wooden cabinet, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl next to the cookie box, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl next to the plate, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl next to the ramekin, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on the cookie box, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on the ramekin, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on the stove, if it is not too much trouble for you, and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl on the wooden cabinet, if it is not too much trouble for you, and place it on the plate",
}

LIBERO_SPATIAL_LENGTH_CONTROL_SUFFIX_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl between the plate and the ramekin and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl from table center and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl next to the cookie box and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl next to the plate and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl next to the ramekin and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on the cookie box and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on the ramekin and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on the stove and place it on the plate; thank you so very much in advance for your help with this",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl on the wooden cabinet and place it on the plate; thank you so very much in advance for your help with this",
}


# ---------------------------------------------------------------------------
# Split 4b same-cue paraphrase controls (the matrix's missing diagonal): the
# target is described with its OWN native cue type, exactly truthfully, but in
# words that differ from the fine-tuning template. 4b only measured cross-cue
# rephrasings (surface -> landmark, * -> region); these separate "the cue type
# changed" from "any wording change at all hurts". Distractor never mentioned,
# scene/init states identical to `libero_spatial`, all 10 tasks.
#   - LEXICAL:   one synonym swap or argument reorder inside the target phrase
#                ("next to" -> "beside", "on" -> "on top of", "plate and
#                ramekin" -> "ramekin and plate", ...). Note "on top of X" is
#                also how negative_contrast rewords tasks 3/5/9's target, so
#                this condition also bounds how much of that condition's drop
#                comes from the target rewording rather than the added clause.
#   - SYNTACTIC: native words kept, target phrase turned into a relative
#                clause ("the black bowl that is <native where>"). Task 2's
#                native "from table center" becomes "that is at table center",
#                the only change beyond inserting "that is".
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_PARAPHRASE_LEXICAL_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl between the ramekin and the plate and place it on the plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl from the middle of the table and place it on the plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl inside the top drawer of the wooden cabinet and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl beside the cookie box and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl beside the plate and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl beside the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on top of the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on top of the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on top of the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl on top of the wooden cabinet and place it on the plate",
}

LIBERO_SPATIAL_PARAPHRASE_SYNTACTIC_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl that is between the plate and the ramekin and place it on the plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":
        "pick up the black bowl that is at table center and place it on the plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl that is in the top drawer of the wooden cabinet and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl that is next to the cookie box and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl that is next to the plate and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl that is next to the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl that is on the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl that is on the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl that is on the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl that is on the wooden cabinet and place it on the plate",
}


# ---------------------------------------------------------------------------
# Split 4b region-cue paraphrase variants: same 8 tasks and the same table-zone
# direction as LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS above (so they
# inherit its truthfulness), reworded so 4b's region-cue result isn't carried
# by one hand-written phrasing. V2 = "<zone> area/part of the table"; V3 =
# side-first reordering ("on the left side of the table, toward the back").
# Run with --task_ids 0 1 3 5 6 7 8 9, same as `grounding/target_cue_region`.
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_TARGET_CUE_REGION_V2_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl in the back part of the table, slightly left of the middle, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl in the back-left corner area of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl around the middle of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl in the back-left area of the table and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl in the front-right area of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl in the front-left area of the table and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl at the very back of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl in the front part of the table, slightly right of the middle, and place it on the plate",
}

LIBERO_SPATIAL_TARGET_CUE_REGION_V3_INSTRUCTIONS = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl toward the back of the table, a little to the left, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on the left side of the table, all the way at the back, and place it on the plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl roughly in the middle of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl on the left side of the table, toward the back, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl on the right side of the table, toward the front, and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl on the left side of the table, toward the front, and place it on the plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":
        "pick up the black bowl all the way at the back of the table and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl toward the front of the table, a little to the right, and place it on the plate",
}


# ---------------------------------------------------------------------------
# Split 4c novel-proximity synonym variants: same 4 surface-family tasks and
# same disclosed-approximate reading as TARGET_CUE_PROXIMITY_NOVEL ("close to
# X"), with three more proximity words that also appear in none of the 10
# native prompts -- so 4c's familiar-vs-novel gap rests on four novel synonyms
# instead of one. Run with --task_ids 3 5 7 9.
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_BESIDE_INSTRUCTIONS = {
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl beside the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl beside the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl beside the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl beside the wooden cabinet and place it on the plate",
}

LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NEAR_INSTRUCTIONS = {
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl near the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl near the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl near the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl near the wooden cabinet and place it on the plate",
}

LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_ADJACENT_INSTRUCTIONS = {
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":
        "pick up the black bowl adjacent to the cookie box and place it on the plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":
        "pick up the black bowl adjacent to the ramekin and place it on the plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":
        "pick up the black bowl adjacent to the stove and place it on the plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":
        "pick up the black bowl adjacent to the wooden cabinet and place it on the plate",
}
