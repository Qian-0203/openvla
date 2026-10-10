"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def retarget_goal_to_distractor(env, target="akita_black_bowl_1", distractor="akita_black_bowl_2"):
    """Swap target and distractor in the task's goal predicates, so success means placing the *other* bowl.

    For language-stress screening conditions whose instruction names the distractor bowl (vla_ws
    screening/language_stress, `"swap_target": true`). The scene and init states are untouched; LIBERO
    parses the goal once at env construction and `step()`'s done flag evaluates it, so this must be
    called again after every env (re)creation. Only flat binary goals are supported, which covers every
    libero_spatial task: `(On akita_black_bowl_1 plate_1)`.
    """
    problem = env.env.parsed_problem
    names = str(problem["objects"])
    assert target in names and distractor in names, f"Expected {target} and {distractor} in the scene: {names}"
    swap = {target: distractor, distractor: target}
    new_goal = []
    for state in problem["goal_state"]:
        assert len(state) == 3 and all(isinstance(tok, str) for tok in state), f"Unsupported goal predicate: {state}"
        new_goal.append([swap.get(tok, tok) for tok in state])
    assert new_goal != problem["goal_state"], f"Goal does not mention {target}: {problem['goal_state']}"
    problem["goal_state"] = new_goal
    return new_goal


class RolloutOutcomeTracker:
    """Env-side record of *which* bowl a rollout reached for, lifted and delivered (model-agnostic).

    Every failure runs to the step limit, so `success` alone cannot tell a wrong-bowl delivery from a
    wrong-bowl grasp or a policy that never commits. This reads the sim state each policy step and
    adds these fields to the results row (all bowl names are BDDL object names, e.g. akita_black_bowl_2):

      target_object / distractor_objects  the goal's first argument (after any swap_target retarget) and
                                          the other black bowls in the scene
      bowls_on_plate                      bowls for which the goal's own predicate (`On(bowl, plate_1)`,
                                          evaluated by LIBERO's `_eval_predicate`) holds at episode end
      first_bowl_approached (+ _step)     first bowl whose center came within APPROACH_THRESH_M (8 cm, 3D)
                                          of the end effector; the same definition, timing (observation
                                          before each policy action) and tie-break as
                                          probe_bowl_attraction.py, so numbers compare with result doc §8.5
      first_bowl_grasped (+ first_grasp_step)
                                          first bowl that rose more than LIFT_THRESH_M (3 cm) above its
                                          post-settle height *while touching the gripper*; the contact
                                          requirement keeps a bowl knocked or tipped by the arm from counting
      approach_step_by_bowl, grasp_step_by_bowl, min_dist_by_bowl
                                          per-bowl detail behind the "first" fields

    Steps are the eval loop's `t` (it includes the num_steps_wait settle steps). Construct after the env
    (and any retarget_goal_to_distractor) exists, call `start_episode()` after each reset, `update(obs, t)`
    at every policy step before the action, and `result()` once the episode ends (success, timeout or
    exception). If the goal is not a flat binary predicate on a tracked object, every field is None.
    """

    APPROACH_THRESH_M = 0.08
    LIFT_THRESH_M = 0.03
    FIELDS = (
        "target_object", "distractor_objects", "bowls_on_plate",
        "first_bowl_approached", "first_bowl_approached_step",
        "first_bowl_grasped", "first_grasp_step",
        "approach_step_by_bowl", "grasp_step_by_bowl", "min_dist_by_bowl",
    )

    def __init__(self, env):
        self.base = env.env  # underlying LIBERO problem env (same access pattern as probe_bowl_attraction.py)
        goal = self.base.parsed_problem["goal_state"]
        bowls = sorted(n for n in self.base.obj_body_id if n.startswith("akita_black_bowl"))
        self.supported = len(goal) == 1 and len(goal[0]) == 3 and goal[0][1] in self.base.obj_body_id
        if self.supported:
            self.predicate, self.target, self.receptacle = goal[0]
            self.objects = sorted(set(bowls) | {self.target})
        self.start_episode()

    def start_episode(self):
        if not self.supported:
            return
        self.settle_z = None  # set at the first policy step, after the sim has dropped the objects
        self.min_dist = {n: float("inf") for n in self.objects}
        self.approach_step = {n: None for n in self.objects}
        self.grasp_step = {n: None for n in self.objects}

    def _pos(self, name):
        return np.array(self.base.sim.data.body_xpos[self.base.obj_body_id[name]])

    def update(self, obs, t):
        if not self.supported:
            return
        if self.settle_z is None:
            self.settle_z = {n: float(self._pos(n)[2]) for n in self.objects}
        eef_pos = np.array(obs["robot0_eef_pos"])
        for n in self.objects:
            pos = self._pos(n)
            d = float(np.linalg.norm(eef_pos - pos))
            self.min_dist[n] = min(self.min_dist[n], d)
            if d < self.APPROACH_THRESH_M and self.approach_step[n] is None:
                self.approach_step[n] = t
            if (
                self.grasp_step[n] is None
                and pos[2] - self.settle_z[n] > self.LIFT_THRESH_M
                and self.base.check_contact(self.base.robots[0].gripper, self.base.get_object(n))
            ):
                self.grasp_step[n] = t

    @staticmethod
    def _first(steps):
        # Earliest step wins; ties break by name, matching probe_bowl_attraction.py's sort.
        hits = sorted((step, n) for n, step in steps.items() if step is not None)
        return hits[0] if hits else (None, None)

    def result(self):
        if not self.supported:
            return {k: None for k in self.FIELDS}
        try:
            on_plate = [
                n for n in self.objects
                if self.base._eval_predicate([self.predicate, n, self.receptacle])
            ]
        except Exception:  # sim left unusable by an exception mid-episode
            on_plate = None
        approach_step, approached = self._first(self.approach_step)
        grasp_step, grasped = self._first(self.grasp_step)
        return {
            "target_object": self.target,
            "distractor_objects": [n for n in self.objects if n != self.target],
            "bowls_on_plate": on_plate,
            "first_bowl_approached": approached,
            "first_bowl_approached_step": approach_step,
            "first_bowl_grasped": grasped,
            "first_grasp_step": grasp_step,
            "approach_step_by_bowl": dict(self.approach_step),
            "grasp_step_by_bowl": dict(self.grasp_step),
            "min_dist_by_bowl": {n: (None if math.isinf(d) else round(d, 4)) for n, d in self.min_dist.items()},
        }


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it.

    `resize_size=None` returns the rotated raw render untouched (no JPEG round-trip, no resize),
    for models that do their own preprocessing (model_family=qwenvla).
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    if resize_size is None:
        return np.ascontiguousarray(img)
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = resize_image(img, resize_size)
    return img


def get_libero_images_openpi(obs, resize_size):
    """Agentview and wrist images preprocessed the way openpi's LIBERO policies were trained.

    Rotated 180 degrees like `get_libero_image`, but resized with padding and without the JPEG
    round trip (openpi's examples/libero/main.py).
    """
    from openpi_client import image_tools  # only needed for openpi runs

    images = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        img = np.ascontiguousarray(obs[key][::-1, ::-1])
        images.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size)))
    return tuple(images)


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
