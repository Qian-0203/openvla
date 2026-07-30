"""
explicit_instructions.py

Explicit, distractor-aware language instructions for the LIBERO Spatial suite.

Each scene contains TWO identical black bowls: the target (`akita_black_bowl_1`,
the object of interest) and a distractor (`akita_black_bowl_2`). The default LIBERO
instructions only describe the target ("pick up the black bowl <where> ..."). The
versions below additionally name where the distractor is ("..., not the one <where>, ...")
to test/teach explicit disambiguation.

Keyed by `task.name` (the BDDL filename without extension). Phrased as a single
imperative clause so they slot cleanly into the OpenVLA prompt template
    "In: What action should the robot take to {instruction.lower()}?\nOut:"
without any template changes. The distractor clause in each entry was verified
against the second bowl's actual init region in the corresponding BDDL file.
"""

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
        "pick up the black bowl at the center of the table, not the one behind it toward the ramekin, and place it on the plate",
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
