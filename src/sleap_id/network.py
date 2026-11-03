"""The identity network: a standard ResNet18 that maps each crop to a point.

The architecture is idtracker.ai's contrastive ResNet18
(idtrackerai/base/network/models.py): torchvision's ResNet18, its first
convolution rebuilt for the crop's channel count, and its last layer replaced by
a bias-free linear map to `embedding_dim` coordinates (only distances between
points matter, so a bias would do nothing). Weights saved by idtracker.ai load
into it unchanged.

The input shape is not fixed by the network: ResNet18 averages over space
before its last layer, so it accepts any crop height and width. The shape is
taken from crops.h5 (what `prepare` computed for this video) and recorded in
`spec()`, so a checkpoint can rebuild the exact network without the config.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet

ARCHITECTURE = "resnet18"
EMBEDDING_DIM = 8           # idtracker.ai's default
MIN_SIDE = 32               # ResNet18 halves the image 5 times


class Embedder(ResNet):
    """ResNet18 from (C, H, W) crops to `embedding_dim` coordinates."""

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int = EMBEDDING_DIM):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        c, h, w = (int(v) for v in input_shape)
        if min(h, w) < MIN_SIDE:
            raise ValueError(f"crops of {h}x{w} are too small for ResNet18 "
                             f"(each side must be >= {MIN_SIDE} pixels)")
        if c != 3:
            self.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.fc = nn.Linear(512 * BasicBlock.expansion, embedding_dim, bias=False)
        self.input_shape = (c, h, w)
        self.embedding_dim = int(embedding_dim)

    def spec(self) -> dict:
        """What a checkpoint needs to rebuild this network."""
        return {"architecture": ARCHITECTURE, "input_shape": list(self.input_shape),
                "embedding_dim": self.embedding_dim}

    @classmethod
    def from_spec(cls, spec: dict) -> "Embedder":
        if spec["architecture"] != ARCHITECTURE:
            raise ValueError(f"unknown architecture {spec['architecture']!r}")
        return cls(tuple(spec["input_shape"]), spec["embedding_dim"])

    def to_input(self, crops: np.ndarray, device: torch.device | str) -> Tensor:
        """uint8 crops (n, H, W) or (n, H, W, 3) -> float (n, C, H, W) in [0, 1]."""
        x = torch.from_numpy(np.ascontiguousarray(crops)).to(device)
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"crops have shape {tuple(x.shape[1:])}, the network "
                             f"expects {self.input_shape}")
        return x.float() / 255


def input_shape_of(crops_path: Path) -> tuple[int, int, int]:
    """(C, H, W) of the crops that `prepare` wrote."""
    with h5py.File(crops_path, "r") as f:
        shape = f["crops"].shape
    return (1,) + shape[1:3] if len(shape) == 3 else (shape[3],) + shape[1:3]


def for_crops(crops_path: Path, embedding_dim: int = EMBEDDING_DIM) -> Embedder:
    return Embedder(input_shape_of(crops_path), embedding_dim)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
