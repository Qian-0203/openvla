"""Utils for evaluating a QwenVLA policy (Qwen3-VL backbone + discrete action tokens).

QwenVLA is not an HF OpenVLA checkpoint: it is a plain `.pt` state dict loaded on top of
`Qwen/Qwen3-VL-8B-Instruct`, and it needs torch>=2.11 / transformers>=5.5, so it runs on the
host in the qwen-vla env rather than in this repo's Docker image.

Rather than re-implement its inference path here, we load `QwenVLAPolicy` straight from the
qwen-vla repo's own `scripts/run_libero_eval.py` -- the exact code path that measured the
checkpoint's libero_spatial success rate. Image preprocessing (raw 256px render -> optional PIL
center crop -> 448px bilinear), the prompt template, action un-normalization, and gripper
handling therefore stay identical to that reference eval.
"""

import importlib.util
import json
import os

import numpy as np


def _load_qwenvla_eval_module(qwenvla_repo):
    path = os.path.join(qwenvla_repo, "scripts", "run_libero_eval.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"QwenVLA eval script not found: {path} (set --qwenvla_repo)")
    # Distinct module name: this repo has its own `run_libero_eval` module.
    spec = importlib.util.spec_from_file_location("qwenvla_run_libero_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_qwenvla_unnorm_key(stats_path, unnorm_key):
    """Same fallback as the OpenVLA path: `libero_spatial` -> `libero_spatial_no_noops`."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    if unnorm_key not in stats and f"{unnorm_key}_no_noops" in stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in stats, f"Action un-norm key {unnorm_key} not found in {stats_path}: {list(stats)}"
    return unnorm_key


def get_qwenvla(cfg):
    """Build a QwenVLAPolicy from `cfg.pretrained_checkpoint` + `cfg.dataset_statistics_path`."""
    assert cfg.qwenvla_repo, "--qwenvla_repo must point at the qwen-vla checkout"
    assert cfg.dataset_statistics_path, "--dataset_statistics_path is required for model_family=qwenvla"
    module = _load_qwenvla_eval_module(cfg.qwenvla_repo)
    unnorm_key = resolve_qwenvla_unnorm_key(cfg.dataset_statistics_path, cfg.unnorm_key)
    policy_cfg = module.GenerateConfig(
        pretrained_checkpoint=str(cfg.pretrained_checkpoint),
        dataset_statistics_path=cfg.dataset_statistics_path,
        unnorm_key=unnorm_key,
        task_suite_name=cfg.task_suite_name,
        action_head="discrete",
        flow_temperature=0.0,  # deterministic decoding, as in the reference eval
        center_crop=cfg.center_crop,
        device="cuda:0",
    )
    policy = module.QwenVLAPolicy(policy_cfg)
    return policy


def get_qwenvla_action(policy, obs, task_label):
    """Return one (7,) action in robosuite convention (gripper already binarized, -1 open / +1 close).

    `obs["full_image"]` must be the 180-degree-rotated raw render (no JPEG round-trip, no resize);
    the policy does its own preprocessing. The prompt is lowercased, matching the OpenVLA path.
    """
    chunk = policy.get_action_chunk(obs["full_image"], task_label.lower())
    return np.asarray(chunk[0], dtype=np.float64)
