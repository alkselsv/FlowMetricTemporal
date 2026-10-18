"""Чтение и запись контрольных точек, общие средства выбора устройства."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import EpisodeClassifier, ModelConfig
from .schema import EPISODE_LABELS, TRAJECTORY_FEATURES

CHECKPOINT_VERSION = 1


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("запрошена CUDA, но она недоступна")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("запрошен MPS, но он недоступен")
    return device


def checkpoint_payload(
    model: EpisodeClassifier,
    *,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_VERSION,
        "labels": list(EPISODE_LABELS),
        "trajectory_features": list(TRAJECTORY_FEATURES),
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[EpisodeClassifier, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("контрольная точка повреждена или имеет неподдерживаемый формат")
    if tuple(payload.get("labels", ())) != EPISODE_LABELS:
        raise ValueError("метки контрольной точки не совпадают с метками среды выполнения")
    if tuple(payload.get("trajectory_features", ())) != TRAJECTORY_FEATURES:
        raise ValueError(
            "траекторные признаки контрольной точки не совпадают с признаками среды выполнения"
        )
    model = EpisodeClassifier(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload
