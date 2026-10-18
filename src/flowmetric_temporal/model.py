"""Shared visual encoder plus trajectory-aware temporal Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .schema import EPISODE_LABELS, TRAJECTORY_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    visual_dim: int = 96
    trajectory_dim: int = 32
    hidden_dim: int = 320
    heads: int = 8
    layers: int = 3
    dropout: float = 0.15
    max_frames: int = 64

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CropEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, output_dim),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        batch, time = crops.shape[:2]
        encoded = self.network(crops.flatten(0, 1))
        return encoded.reshape(batch, time, -1)


class EpisodeClassifier(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.hidden_dim % cfg.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.crop_encoder = CropEncoder(cfg.visual_dim)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(len(TRAJECTORY_FEATURES), cfg.trajectory_dim),
            nn.LayerNorm(cfg.trajectory_dim),
            nn.GELU(),
        )
        fused_dim = 3 * cfg.visual_dim + cfg.trajectory_dim
        if fused_dim != cfg.hidden_dim:
            raise ValueError("hidden_dim must equal 3 * visual_dim + trajectory_dim")
        self.position = nn.Parameter(torch.empty(1, cfg.max_frames, cfg.hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.layers, norm=nn.LayerNorm(cfg.hidden_dim))
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, len(EPISODE_LABELS)))

    def forward(
        self,
        staff: torch.Tensor,
        customer: torch.Tensor,
        table: torch.Tensor,
        trajectory: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        time = staff.shape[1]
        if time > self.config.max_frames:
            raise ValueError(f"episode has {time} frames; maximum is {self.config.max_frames}")
        if mask.dtype != torch.bool or mask.shape != staff.shape[:2]:
            raise ValueError("mask must be bool [batch, time]")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each episode must contain at least one valid frame")
        fused = torch.cat(
            [
                self.crop_encoder(staff),
                self.crop_encoder(customer),
                self.crop_encoder(table),
                self.trajectory_encoder(trajectory),
            ],
            dim=-1,
        )
        encoded = self.temporal(fused + self.position[:, :time], src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.head(pooled)

