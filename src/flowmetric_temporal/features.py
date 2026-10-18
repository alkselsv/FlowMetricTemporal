"""Интерполяция траекторий и нормализованные временные признаки."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import TRAJECTORY_FEATURES


def track_boxes(tracks: dict[str, Any], track_id: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        track = tracks["tracks"][str(track_id)]
        frames = np.asarray(track["frames"], dtype=np.float32)
        boxes = np.asarray(track["boxes"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"трек {track_id!r} отсутствует или имеет неверный формат") from exc
    if frames.ndim != 1 or boxes.shape != (len(frames), 4) or len(frames) == 0:
        raise ValueError(f"трек {track_id!r} содержит несовместимые frames и boxes")
    if np.any(np.diff(frames) <= 0) or not np.isfinite(boxes).all():
        raise ValueError(
            f"трек {track_id!r} должен содержать упорядоченные конечные наблюдения"
        )
    return frames, boxes


def interpolate_boxes(
    observed_frames: np.ndarray,
    observed_boxes: np.ndarray,
    target_frames: np.ndarray,
    *,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Интерполировать ограничивающие прямоугольники и отметить наблюдаемые цели."""
    boxes = np.column_stack(
        [np.interp(target_frames, observed_frames, observed_boxes[:, i]) for i in range(4)]
    ).astype(np.float32)
    right = np.searchsorted(observed_frames, target_frames, side="left")
    left = np.clip(right - 1, 0, len(observed_frames) - 1)
    right = np.clip(right, 0, len(observed_frames) - 1)
    nearest = np.minimum(
        np.abs(target_frames - observed_frames[left]),
        np.abs(target_frames - observed_frames[right]),
    )
    visible = nearest <= max_gap
    return boxes, visible


def _box_geometry(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    return np.column_stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / (2 * width),
            (boxes[:, 1] + boxes[:, 3]) / (2 * height),
            (boxes[:, 2] - boxes[:, 0]) / width,
            (boxes[:, 3] - boxes[:, 1]) / height,
        ]
    ).astype(np.float32)


def trajectory_features(
    staff_boxes: np.ndarray,
    customer_boxes: np.ndarray,
    table_box: np.ndarray | None,
    visible: np.ndarray,
    *,
    width: int,
    height: int,
    fps: float,
    frames: np.ndarray,
) -> np.ndarray:
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("ширина, высота и частота кадров должны быть положительными")
    if staff_boxes.shape != customer_boxes.shape or staff_boxes.shape != (len(frames), 4):
        raise ValueError(
            "ограничивающие прямоугольники людей должны иметь форму [время, 4]"
        )
    staff = _box_geometry(staff_boxes, width, height)
    customer = _box_geometry(customer_boxes, width, height)
    if table_box is None:
        table = np.zeros_like(staff)
    else:
        raw_table = np.broadcast_to(np.asarray(table_box, dtype=np.float32), staff_boxes.shape)
        table = _box_geometry(raw_table, width, height)

    delta = customer[:, :2] - staff[:, :2]
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    times = frames.astype(np.float32) / float(fps)
    dt = np.diff(times)
    staff_velocity = np.zeros((len(frames), 2), dtype=np.float32)
    customer_velocity = np.zeros_like(staff_velocity)
    if len(frames) > 1:
        denom = np.maximum(dt[:, None], 1e-6)
        staff_velocity[1:] = np.diff(staff[:, :2], axis=0) / denom
        customer_velocity[1:] = np.diff(customer[:, :2], axis=0) / denom
        staff_velocity[0] = staff_velocity[1]
        customer_velocity[0] = customer_velocity[1]
    result = np.column_stack(
        [staff, customer, table, delta, distance, staff_velocity, customer_velocity, visible]
    ).astype(np.float32)
    if result.shape[1] != len(TRAJECTORY_FEATURES):
        raise AssertionError("контракт траекторных признаков нарушен")
    return result
