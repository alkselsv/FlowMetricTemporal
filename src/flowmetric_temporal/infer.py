"""Batch inference for extracted episode manifests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import EpisodeDataset
from .runtime import load_checkpoint, resolve_device
from .schema import EPISODE_LABELS


@torch.inference_mode()
def predict_manifest(
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str | None = None,
    batch_size: int = 16,
    device_name: str = "auto",
) -> None:
    device = resolve_device(device_name)
    model, _ = load_checkpoint(checkpoint, device)
    loader = DataLoader(EpisodeDataset(manifest, split=split), batch_size=batch_size, shuffle=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for batch in loader:
        logits = model(
            batch["staff"].to(device),
            batch["customer"].to(device),
            batch["table"].to(device),
            batch["trajectory"].to(device),
            batch["mask"].to(device),
        )
        probabilities = logits.softmax(dim=1).cpu()
        for episode_id, scores in zip(batch["episode_id"], probabilities):
            prediction = int(scores.argmax())
            rows.append(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "label": EPISODE_LABELS[prediction],
                        "confidence": float(scores[prediction]),
                        "probabilities": {
                            label: float(score) for label, score in zip(EPISODE_LABELS, scores)
                        },
                    }
                )
            )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")

