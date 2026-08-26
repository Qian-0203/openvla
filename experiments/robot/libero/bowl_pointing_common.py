"""
bowl_pointing_common.py

Shared rendering/annotation/scoring logic for the bowl-pointing VQA probe (see
probe_bowl_pointing.py's module docstring for the full rationale). Deliberately has NO dependency
on `experiments.robot.openvla_utils` or any specific model's `transformers` version -- probing a
different VLM (e.g. probe_bowl_pointing_qwen.py) commonly needs a newer `transformers` than the
pinned OpenVLA eval image ships, and this module must stay importable either way since it only
touches LIBERO/robosuite, never a model.
"""

import re

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.robot.libero.instructions import (
    LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS,
    LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS,
    LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS,
)
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, resize_image

# condition -> (task_suite_name, instruction dict). negative_contrast/positive_contrast/default
# share the plain 2-bowl `libero_spatial` scene (only the prompt differs); hardneg uses the 3-bowl
# hard-negative scene. Rendering is cached per (suite, task_id) so the shared scene is only
# rendered once even though multiple conditions use it. instruction dict of None means "use
# LIBERO's own native task language" (the distractor is never mentioned at all) -- the same
# convention eval_registry.CONDITIONS uses for the real-eval `default` condition.
CONDITION_SUITES = {
    "default": ("libero_spatial", None),
    "negative_contrast": ("libero_spatial", LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS),
    "positive_contrast": ("libero_spatial", LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS),
    "hardneg": ("libero_spatial_3bowl_hardneg", LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS),
    "hardneg_default": ("libero_spatial_3bowl_hardneg", None),  # same 3-bowl scene as hardneg, but
    # LIBERO's own native (target-only) task language -- the distractor family is never mentioned.
    # Comparison point for `hardneg` the same way `default` is for negative_contrast/positive_contrast.
}

MARKER_COLORS = ["#e6194b", "#3cb44b", "#4363d8"]  # red / green / blue -- distinct on a wood table


def _bowl_pixel_centroid(base, name, resolution, seg):
    """Ground-truth (row, col) of `name`'s rendered pixels in `seg`'s frame, from MuJoCo's own
    segmentation render -- not a hand-computed 3D->2D projection. `project_points_from_world_to_
    camera` was found to silently mis-project some (but not all) objects in a scene by tens of
    pixels (see benchmark_split_result.md Sec.8 marker-alignment investigation, 2026-08-26); reading
    back which pixels the renderer itself assigned to the body's geoms sidesteps that whole class of
    bug, since segmentation and the RGB frame come from the identical draw call.
    """
    body_id = base.obj_body_id[name]
    geom_ids = [i for i in range(base.sim.model.ngeom) if base.sim.model.geom_bodyid[i] == body_id]
    mask = np.isin(seg[:, :, 1], geom_ids) & (seg[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM))
    rows, cols = np.where(mask)
    assert len(rows) > 0, f"body '{name}' has 0 visible pixels in the agentview render -- fully occluded?"
    return rows.mean(), cols.mean()


def render_and_annotate(task_suite, task_id, resolution, resize_size, num_steps_wait, rng):
    """Renders episode-0's init state and overlays a shuffled number on each black bowl.

    Returns (annotated_pil_image, bowl_to_number dict, target_number, raw_unannotated_pil_image,
    default_task_description) -- the last is LIBERO's own native task language (target-only, no
    distractor mention), used by the `default` condition.
    """
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)

    env, default_task_description = get_libero_env(task, "openvla", resolution=resolution)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    base = env.env  # underlying robosuite/BDDL env (per LIBERO/scripts/render_2v3_bowl_init.py)
    bowls = sorted(n for n in base.obj_body_id if n.startswith("akita_black_bowl"))
    assert "akita_black_bowl_1" in bowls, f"task {task.name} has no akita_black_bowl_1 (target)"

    # Same camera/resolution the agentview obs was rendered with, so segmentation pixels line up
    # 1:1 with obs["agentview_image"] before either gets flipped.
    seg = base.sim.render(width=resolution, height=resolution, camera_name="agentview", segmentation=True)
    centroids_rc = [_bowl_pixel_centroid(base, n, resolution, seg) for n in bowls]
    env.close()

    # Raw obs image (and the raw segmentation render above) are in the same unflipped orientation.
    # get_libero_image() flips both axes (180-degree rotation) to match training preprocessing --
    # apply the identical transform to both the image and the marker centroids.
    raw_img = obs["agentview_image"]
    flipped = raw_img[::-1, ::-1]
    flipped_resized = resize_image(flipped, (resize_size, resize_size))  # same helper get_libero_image() uses
    raw_pil = Image.fromarray(flipped_resized).convert("RGB")

    scale = resize_size / resolution
    numbers = list(range(1, len(bowls) + 1))
    rng.shuffle(numbers)
    bowl_to_number = dict(zip(bowls, numbers))

    annotated = raw_pil.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=16)
    radius = 6  # small outline dot -- big enough to see, small enough not to occlude the bowl
    for name, (row, col) in zip(bowls, centroids_rc):
        flipped_row = resolution - 1 - row
        flipped_col = resolution - 1 - col
        x, y = flipped_col * scale, flipped_row * scale
        color = MARKER_COLORS[(bowl_to_number[name] - 1) % len(MARKER_COLORS)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.line((x - 3, y, x + 3, y), fill=color, width=1)
        draw.line((x, y - 3, x, y + 3), fill=color, width=1)
        # Label sits offset above-right of the dot, with a white halo so it stays legible on both
        # light and dark table regions, instead of covering the bowl the way a filled circle did.
        label = str(bowl_to_number[name])
        lx, ly = x + radius + 2, y - radius - 14
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((lx + dx, ly + dy), label, fill="white", font=font)
        draw.text((lx, ly), label, fill=color, font=font)

    target_number = bowl_to_number["akita_black_bowl_1"]
    return annotated, bowl_to_number, target_number, raw_pil, default_task_description


def parse_answer(raw_text, valid_numbers):
    """First integer in raw_text that's one of the valid marker numbers; None if none found."""
    for match in re.finditer(r"\d+", raw_text):
        n = int(match.group())
        if n in valid_numbers:
            return n
    return None
