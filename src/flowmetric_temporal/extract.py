"""Построение компактных тензоров эпизодов из видео, треков и аннотаций."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .features import interpolate_boxes, track_boxes, trajectory_features
from .schema import EPISODE_LABELS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: некорректный JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: строка должна содержать объект")
        rows.append(row)
    return rows


def sample_frames(start: int, end: int, count: int) -> np.ndarray:
    if start < 0 or end < start or count < 2:
        raise ValueError("некорректный интервал эпизода или число кадров")
    return np.rint(np.linspace(start, end, count)).astype(np.int64)


def _expand_box(box: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    cx, cy = (box[:2] + box[2:]) / 2
    bw, bh = (box[2:] - box[:2]) * scale
    return np.asarray(
        [max(0, cx - bw / 2), max(0, cy - bh / 2), min(width, cx + bw / 2), min(height, cy + bh / 2)],
        dtype=np.float32,
    )


def crop_resize(frame: np.ndarray, box: np.ndarray | None, size: int) -> np.ndarray:
    if box is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    clipped = _expand_box(np.asarray(box, dtype=np.float32), 1.15, w, h)
    x1, y1, x2, y2 = np.rint(clipped).astype(int)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def load_video_frames(path: Path, frame_indices: Iterable[int]) -> dict[int, np.ndarray]:
    requested = sorted(set(int(value) for value in frame_indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"не удалось открыть видео: {path}")
    output: dict[int, np.ndarray] = {}
    try:
        for index in requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"не удалось прочитать кадр {index} из {path}")
            output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return output


def extract_episode(
    annotation: dict[str, Any],
    *,
    frames_per_episode: int,
    crop_size: int,
) -> dict[str, np.ndarray]:
    required = {"episode_id", "label", "video", "tracks", "start_frame", "end_frame", "staff_id", "customer_id"}
    missing = required - annotation.keys()
    if missing:
        raise ValueError(f"в аннотации отсутствуют поля: {', '.join(sorted(missing))}")
    if annotation["label"] not in EPISODE_LABELS:
        raise ValueError(f"неизвестная метка {annotation['label']!r}")
    tracks_path = Path(annotation["tracks"])
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    fps = float(tracks["fps"])
    width, height = int(tracks["width"]), int(tracks["height"])
    indices = sample_frames(int(annotation["start_frame"]), int(annotation["end_frame"]), frames_per_episode)
    max_gap = max(1.0, fps * 0.5)
    sf, sb = track_boxes(tracks, str(annotation["staff_id"]))
    cf, cb = track_boxes(tracks, str(annotation["customer_id"]))
    staff_boxes, staff_visible = interpolate_boxes(sf, sb, indices, max_gap=max_gap)
    customer_boxes, customer_visible = interpolate_boxes(cf, cb, indices, max_gap=max_gap)
    visible = staff_visible & customer_visible
    table_box = annotation.get("table_box")
    table = None if table_box is None else np.asarray(table_box, dtype=np.float32)
    if table is not None and (table.shape != (4,) or table[0] >= table[2] or table[1] >= table[3]):
        raise ValueError("table_box должен содержать упорядоченные координаты xyxy")
    video_frames = load_video_frames(Path(annotation["video"]), indices)
    staff_crops = []
    customer_crops = []
    table_crops = []
    for i, frame_index in enumerate(indices):
        frame = video_frames[int(frame_index)]
        staff_crops.append(crop_resize(frame, staff_boxes[i] if staff_visible[i] else None, crop_size))
        customer_crops.append(crop_resize(frame, customer_boxes[i] if customer_visible[i] else None, crop_size))
        table_crops.append(crop_resize(frame, table, crop_size))
    return {
        "staff": np.stack(staff_crops),
        "customer": np.stack(customer_crops),
        "table": np.stack(table_crops),
        "trajectory": trajectory_features(
            staff_boxes,
            customer_boxes,
            table,
            visible,
            width=width,
            height=height,
            fps=fps,
            frames=indices,
        ),
        "mask": visible.astype(np.bool_),
        "frames": indices,
    }


def build_dataset(
    annotations_path: Path,
    output_dir: Path,
    *,
    frames_per_episode: int = 16,
    crop_size: int = 112,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_rows: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(annotations_path):
        episode_id = str(row.get("episode_id", "")).strip()
        if not episode_id or episode_id in seen or "/" in episode_id or "\\" in episode_id:
            raise ValueError(
                f"episode_id должен быть уникальным и пригодным для имени файла: {episode_id!r}"
            )
        seen.add(episode_id)
        arrays = extract_episode(row, frames_per_episode=frames_per_episode, crop_size=crop_size)
        filename = f"{episode_id}.npz"
        np.savez_compressed(output_dir / filename, **arrays)
        manifest_rows.append(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "label": row["label"],
                    "split": row.get("split", "train"),
                    "npz": filename,
                },
                ensure_ascii=False,
            )
        )
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return manifest_path
