"""Utils for evaluating OpenVLA-OFT checkpoints (e.g. moojink/openvla-7b-oft-finetuned-libero-spatial).

An OFT checkpoint is a merged OpenVLA model plus two small side modules that sit next to the weights:
a continuous L1-regression action head (`action_head--*_checkpoint.pt`) and a proprio projector
(`proprio_projector--*_checkpoint.pt`). The policy takes two images (agentview + wrist) and the 8-d proprio
state, and predicts a chunk of `NUM_ACTIONS_CHUNK` (8) actions in parallel, which are executed open loop.

Adapted from moojink/openvla-oft (MIT): experiments/robot/openvla_utils.py and
prismatic/models/{action_heads,projectors}.py. Only the L1 path is kept (no FiLM, no diffusion head), which
is what the released LIBERO checkpoints use. The checkpoint ships its own modeling_prismatic.py; it imports
prismatic.vla.constants and prismatic.training.train_utils, which this repo carries (copied from openvla-oft).
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NormalizationType,
)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return self.ffn(x) + x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)


class L1RegressionActionHead(nn.Module):
    """MLP action head that regresses a chunk of continuous actions (trained with an L1 loss)."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(num_blocks=2, input_dim=input_dim * ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim)

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: (batch, chunk_len * action_dim, hidden_dim) -> (batch, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        return self.model(actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1))


class ProprioProjector(nn.Module):
    """Projects the proprio state into the LLM's embedding space."""

    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim
        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        return self.fc2(self.act_fn1(self.fc1(proprio)))


def _find_component(checkpoint_dir: str, name: str) -> str:
    matches = [f for f in os.listdir(checkpoint_dir) if name in f and "checkpoint" in f]
    assert len(matches) == 1, f"Expected exactly 1 {name} checkpoint in {checkpoint_dir}, found {matches}"
    return os.path.join(checkpoint_dir, matches[0])


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    # Components trained under DDP carry a "module." prefix.
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def resize_image_for_policy(img: np.ndarray, size: int = OPENVLA_IMAGE_SIZE) -> np.ndarray:
    """Same resize as the training data pipeline: JPEG round trip, then Lanczos3 (libero_utils.resize_image)."""
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def center_crop_image(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    """Center crop of area `crop_scale`, resized back to 224x224 (matches the random-crop training augmentation)."""
    image = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(image)), tf.float32)
    image = tf.expand_dims(image, axis=0)
    side = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
    offset = (1 - side) / 2
    boxes = tf.reshape(tf.stack([offset, offset, offset + side, offset + side]), (1, 4))
    image = tf.image.crop_and_resize(image, boxes, tf.range(1), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE))[0]
    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """Normalize proprio the way the checkpoint was trained (bounds_q99 for LIBERO)."""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        high, low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        high, low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    return np.clip(np.where(mask, 2 * (proprio - low) / (high - low + 1e-8) - 1, proprio), a_min=-1.0, a_max=1.0)


class OFTPolicy:
    """An OFT checkpoint: VLA, processor, L1 action head and proprio projector, loaded from one directory."""

    def __init__(self, cfg):
        ckpt = str(cfg.pretrained_checkpoint)
        assert os.path.isdir(ckpt), f"OFT checkpoints are loaded from a local directory, got {ckpt!r}"
        attn_implementation = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
        # The checkpoint's auto_map points at its own configuration_/modeling_prismatic.py (remote code).
        self.vla = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.vla.vision_backbone.set_num_images_in_input(2)  # agentview + wrist
        self.vla.eval()
        self.vla = self.vla.to(DEVICE)

        stats_path = os.path.join(ckpt, "dataset_statistics.json")
        assert os.path.isfile(stats_path), f"{stats_path} is needed for action un-normalization"
        with open(stats_path, "r") as f:
            self.vla.norm_stats = json.load(f)

        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

        llm_dim = self.vla.llm_dim
        self.action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
        self.action_head.load_state_dict(_load_state_dict(_find_component(ckpt, "action_head")))
        self.action_head = self.action_head.to(torch.bfloat16).to(DEVICE).eval()

        self.proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
        self.proprio_projector.load_state_dict(_load_state_dict(_find_component(ckpt, "proprio_projector")))
        self.proprio_projector = self.proprio_projector.to(torch.bfloat16).to(DEVICE).eval()

        self.center_crop = cfg.center_crop

    @property
    def norm_stats(self):
        return self.vla.norm_stats

    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3, "expected an HxWx3 uint8 image"
        if image.shape[:2] != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
            image = resize_image_for_policy(image)
        pil = Image.fromarray(image).convert("RGB")
        return center_crop_image(pil) if self.center_crop else pil

    @torch.inference_mode()
    def predict_chunk(self, obs: Dict[str, Any], task_label: str, unnorm_key: str) -> List[np.ndarray]:
        """Returns `NUM_ACTIONS_CHUNK` un-normalized actions for one observation.

        obs: {"full_image", "wrist_image"}: 180-degree-rotated, 224x224 uint8; "state": 8-d proprio.
        """
        primary, wrist = self._prepare_image(obs["full_image"]), self._prepare_image(obs["wrist_image"])
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = self.processor(prompt, primary).to(DEVICE, dtype=torch.bfloat16)
        wrist_inputs = self.processor(prompt, wrist).to(DEVICE, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

        proprio = normalize_proprio(np.asarray(obs["state"], dtype=np.float64), self.vla.norm_stats[unnorm_key]["proprio"])
        actions, _ = self.vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=None,
            action_head=self.action_head,
            use_film=False,
        )
        return [actions[i] for i in range(len(actions))]
