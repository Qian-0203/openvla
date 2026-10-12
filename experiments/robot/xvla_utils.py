"""Client side of the X-VLA LIBERO protocol (model_family=xvla).

Adapted from 2toINF/X-VLA (Apache-2.0), evaluation/libero/libero_client.py. X-VLA is evaluated differently
from the OpenVLA-style policies in this repo, and the pieces that differ live here and in two places of
run_libero_eval.py:

- it commands *absolute* end-effector poses: after the settle steps the OSC controller is switched to
  `use_delta = False`, and each action is [pos(3), axis-angle(3), gripper(1)] in the robot base frame;
- the policy predicts 30 actions as [pos(3), rotation-6D(6), gripper(1)] and all 30 are executed before the next
  query; the 6D rotation is converted to axis-angle here, the gripper is binarized at 0.5 to {-1, +1};
- its proprio input is the closed-loop pose only for the first query of an episode; afterwards it is the last
  *predicted* absolute pose of the previous chunk (so the client carries state, the server stays stateless);
- the agentview image is rotated 180 degrees, the wrist image is NOT, and both are sent at 256x256;
- the model itself runs in a policy server (scripts/policy_servers/serve_xvla.py in vla_ws), because it needs
  its own pinned transformers / timm. The wire format is the openpi websocket protocol.
"""

import collections
from typing import Any, Dict

import numpy as np
import robosuite.utils.transform_utils as T

EPS = 1e-6


def mat_to_rotate6d(R: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> its first two columns, flattened (6,)."""
    return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)


def rotate6d_to_axisangle(r6d: np.ndarray) -> np.ndarray:
    """(N, 6) rotation-6D (Zhou et al.) -> (N, 3) axis-angle, same construction as X-VLA's client."""
    a1, a2 = r6d[:, 0:3], r6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    b2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)
    return np.stack([T.quat2axisangle(T.mat2quat(R[i])) for i in range(R.shape[0])], axis=0)


class XVLAPolicy:
    """Stateful client: carries the proprio chain and the queued actions of the current episode."""

    def __init__(self, client: Any, domain_id: int = 3, steps: int = 10):
        self.client = client
        self.domain_id = int(domain_id)
        self.steps = int(steps)
        self.reset()

    def get_server_metadata(self) -> Dict[str, Any]:
        return self.client.get_server_metadata()

    def reset(self) -> None:
        self.proprio = None
        self.plan = collections.deque()

    def act(self, obs: Dict[str, Any], controller: Any, prompt: str) -> np.ndarray:
        """One environment action (7,) in the absolute-pose convention; queries the server every 30 steps."""
        if not self.plan:
            closed_loop = np.concatenate(
                [np.asarray(controller.ee_pos), mat_to_rotate6d(np.asarray(controller.ee_ori_mat)), [0.0]], axis=-1
            )
            closed_loop = np.concatenate([closed_loop, np.zeros_like(closed_loop)], axis=-1)
            if self.proprio is None:
                self.proprio = closed_loop
            reply = self.client.infer({
                "image0": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                "image1": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
                "language_instruction": str(prompt),
                "proprio": self.proprio.astype(np.float32),
                "domain_id": self.domain_id,
                "steps": self.steps,
            })
            action = np.asarray(reply["action"], dtype=np.float64)  # (T, 10): pos3, rot6d, grip
            if action.ndim != 2 or action.shape[1] < 10:
                raise RuntimeError(f"unexpected action shape from the X-VLA server: {action.shape}")
            self.proprio[:9] = action[-1, :9]
            final = np.concatenate(
                [action[:, :3], rotate6d_to_axisangle(action[:, 3:9]), action[:, 9:10]], axis=-1
            )
            self.plan.extend(final.tolist())
        a = np.array(self.plan.popleft(), dtype=np.float32)
        a[-1] = 1.0 if a[-1] > 0.5 else -1.0
        return a
