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
