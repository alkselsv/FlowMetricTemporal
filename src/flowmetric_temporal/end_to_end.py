"""Разметка, извлечение, обучение и предсказание одной командой."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_annotations
from .extract import build_dataset
from .infer import predict_manifest
from .train import train_model


def run_current_data(
    flowmetric_root: Path,
    output_dir: Path,
    *,
    tracks_relative: Path = Path("data/output/service_verifier_tables_run/tracks.json"),
    frames_per_episode: int = 16,
    crop_size: int = 112,
    epochs: int = 30,
    batch_size: int = 8,
    device_name: str = "auto",
    prepare_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = output_dir / "annotations.jsonl"
    bootstrap = bootstrap_annotations(
        flowmetric_root,
        annotations,
        tracks_relative=tracks_relative,
    )
    manifest = build_dataset(
        annotations,
        output_dir / "episodes",
        frames_per_episode=frames_per_episode,
        crop_size=crop_size,
    )
    result: dict[str, Any] = {"bootstrap": bootstrap, "manifest": str(manifest)}
    if not prepare_only:
        checkpoint = output_dir / "best.pt"
        metrics = train_model(
            manifest,
            checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            device_name=device_name,
        )
        predictions = output_dir / "val_predictions.jsonl"
        predict_manifest(
            manifest,
            checkpoint,
            predictions,
            split="val",
            batch_size=batch_size,
            device_name=device_name,
        )
        result.update(
            checkpoint=str(checkpoint),
            predictions=str(predictions),
            validation_metrics=metrics,
        )
    report = output_dir / "run.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["report"] = str(report)
    return result
