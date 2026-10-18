"""Training loop with class weighting and macro-F1 checkpoint selection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import EpisodeDataset
from .model import EpisodeClassifier, ModelConfig
from .runtime import checkpoint_payload, resolve_device
from .schema import EPISODE_LABELS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("metrics require equally sized, non-empty inputs")
    f1_scores: list[float] = []
    for label in range(len(EPISODE_LABELS)):
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        f1_scores.append(2 * tp / denominator if denominator else 0.0)
    accuracy = sum(t == p for t, p in zip(targets, predictions)) / len(targets)
    return {"accuracy": accuracy, "macro_f1": sum(f1_scores) / len(f1_scores)}


def _forward(model: EpisodeClassifier, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return model(
        batch["staff"].to(device),
        batch["customer"].to(device),
        batch["table"].to(device),
        batch["trajectory"].to(device),
        batch["mask"].to(device),
    )


@torch.inference_mode()
def evaluate(model: EpisodeClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for batch in loader:
        logits = _forward(model, batch, device)
        targets.extend(batch["label"].tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return classification_metrics(targets, predictions)


def train_model(
    manifest: Path,
    output: Path,
    *,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
    workers: int = 0,
    seed: int = 17,
    config: ModelConfig | None = None,
) -> dict[str, float]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(seed)
    device = resolve_device(device_name)
    train_data = EpisodeDataset(manifest, split="train")
    val_data = EpisodeDataset(manifest, split="val")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = EpisodeClassifier(config).to(device)
    counts = torch.bincount(
        torch.tensor([EPISODE_LABELS.index(row["label"]) for row in train_data.rows]),
        minlength=len(EPISODE_LABELS),
    ).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_metrics = {"accuracy": 0.0, "macro_f1": -1.0}
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch, device)
            targets = batch["label"].to(device)
            loss = loss_fn(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            samples += len(targets)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = total_loss / max(samples, 1)
        print(json.dumps({"epoch": epoch, **metrics}))
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            torch.save(checkpoint_payload(model, epoch=epoch, metrics=metrics), output)
    return best_metrics

