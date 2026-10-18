"""Dataset for pre-extracted temporal episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import EPISODE_LABELS, TRAJECTORY_FEATURES


class EpisodeDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: str | Path, *, split: str | None = None) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_no} is invalid JSON") from exc
            if row.get("label") not in EPISODE_LABELS:
                raise ValueError(f"manifest line {line_no} has an unknown label")
            if split is None or row.get("split", "train") == split:
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"manifest has no rows for split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.root / row["npz"]
        with np.load(path, allow_pickle=False) as payload:
            result: dict[str, Any] = {}
            time_steps: int | None = None
            for name in ("staff", "customer", "table"):
                array = payload[name]
                if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
                    raise ValueError(f"{path}: {name} must be uint8 [time, height, width, 3]")
                time_steps = len(array) if time_steps is None else time_steps
                if len(array) != time_steps:
                    raise ValueError(f"{path}: crop streams have different lengths")
                result[name] = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            trajectory = payload["trajectory"].astype(np.float32)
            mask = payload["mask"].astype(np.bool_)
            if trajectory.shape != (time_steps, len(TRAJECTORY_FEATURES)):
                raise ValueError(f"{path}: incompatible trajectory shape")
            if mask.shape != (time_steps,) or not mask.any():
                raise ValueError(f"{path}: mask must select at least one time step")
            result["trajectory"] = torch.from_numpy(trajectory)
            result["mask"] = torch.from_numpy(mask)
        result["label"] = torch.tensor(EPISODE_LABELS.index(row["label"]), dtype=torch.long)
        result["episode_id"] = str(row["episode_id"])
        return result

