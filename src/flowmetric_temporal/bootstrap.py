"""Формирование проверяемого набора эпизодов из контрольных данных FlowMetric."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import interpolate_boxes, track_boxes

SHORT_TRACK_PENALTY = 40.0


@dataclass(frozen=True)
class WindowStats:
    staff_id: str
    customer_id: str
    start_frame: int
    end_frame: int
    mean_distance: float
    min_distance: float
    staff_speed: float
    near_table_ratio: float
    table_box: tuple[float, float, float, float] | None


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"не удалось прочитать JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} должен содержать объект JSON")
    return payload


def map_gt_to_tracks(gt: dict[str, Any], tracks: dict[str, Any]) -> dict[int, str]:
    """Сопоставить опорные точки головы с идентификаторами FlowMetric."""
    fps = float(tracks["fps"])
    by_frame: dict[int, list[tuple[str, list[float], float]]] = {}
    for track_id, track in tracks["tracks"].items():
        life = (track["frames"][-1] - track["frames"][0]) / fps
        for frame, box in zip(track["frames"], track["boxes"]):
            by_frame.setdefault(int(frame), []).append((str(track_id), box, life))

    def nested_fragment(
        candidate: tuple[str, list[float], float],
        others: list[tuple[str, list[float], float]],
    ) -> bool:
        x1, y1, x2, y2 = candidate[1]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        for other in others:
            if other is candidate or other[2] < 1.0:
                continue
            ox1, oy1, ox2, oy2 = other[1]
            intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
                0.0, min(y2, oy2) - max(y1, oy1)
            )
            if intersection / area > 0.8:
                return True
        return False

    mapping: dict[int, str] = {}
    used: set[str] = set()
    for frame_key, raw_points in gt.get("anchors", {}).items():
        if not str(frame_key).isdigit() or not isinstance(raw_points, dict):
            continue
        frame_candidates = by_frame.get(int(frame_key), [])
        candidates = [
            candidate
            for candidate in frame_candidates
            if candidate[0] not in used
            and not (candidate[2] < 1.0 and nested_fragment(candidate, frame_candidates))
        ]
        people = [(int(person), point) for person, point in raw_points.items()]
        if not candidates or not people:
            continue
        cost = np.full((len(people), len(candidates)), 1e6)
        for person_index, (_, (x, y)) in enumerate(people):
            for candidate_index, (_, (x1, y1, x2, y2), life) in enumerate(candidates):
                if x1 - 10 <= x <= x2 + 10 and y1 - 30 <= y <= y1 + 0.5 * (y2 - y1):
                    distance = np.hypot(
                        x - (x1 + x2) / 2,
                        y - (y1 + 0.12 * (y2 - y1)),
                    )
                    cost[person_index, candidate_index] = distance + (
                        SHORT_TRACK_PENALTY if life < 1.0 else 0.0
                    )
        for person_index, candidate_index in zip(*linear_sum_assignment(cost)):
            if cost[person_index, candidate_index] < 1e6:
                mapping[people[person_index][0]] = candidates[candidate_index][0]
                used.add(candidates[candidate_index][0])
    return mapping


def ground_truth_people(gt: dict[str, Any]) -> set[int]:
    people: set[int] = set(int(person) for person in gt.get("staff", []))
    for frame, points in gt.get("anchors", {}).items():
        if str(frame).isdigit() and isinstance(points, dict):
            people.update(int(person) for person in points)
    for event in gt.get("events", []):
        for field in ("person", "staff", "customer"):
            if event.get(field) is not None:
                people.add(int(event[field]))
    return people


def service_intervals(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Объединить начала и окончания обслуживания, пропуская нулевые наблюдения."""
    duration = float(gt["duration_s"])
    active: dict[tuple[int, int], float] = {}
    intervals: list[dict[str, Any]] = []
    for event in sorted(gt.get("events", []), key=lambda item: float(item["t"])):
        if event.get("type") not in {"service_start", "service_end"}:
            continue
        pair = (int(event["staff"]), int(event["customer"]))
        time = float(event["t"])
        if event["type"] == "service_start":
            active[pair] = time
        else:
            start = active.pop(pair, 0.0)
            if time > start:
                intervals.append({"staff": pair[0], "customer": pair[1], "start": start, "end": time})
    for (staff, customer), start in active.items():
        if duration > start:
            intervals.append({"staff": staff, "customer": customer, "start": start, "end": duration})
    return intervals


def _table_box(table_config: dict[str, Any], customer_boxes: np.ndarray) -> tuple[float, float, float, float] | None:
    boxes = [np.asarray(row["box"], dtype=np.float32) for row in table_config.get("tables", [])]
    if not boxes:
        return None
    customer_center = np.mean((customer_boxes[:, :2] + customer_boxes[:, 2:]) / 2, axis=0)
    distances = []
    for box in boxes:
        center = (box[:2] + box[2:]) / 2
        distances.append(float(np.linalg.norm(center - customer_center)))
    selected = boxes[int(np.argmin(distances))]
    return tuple(float(value) for value in selected)


def _point_box_distance(point: np.ndarray, box: np.ndarray) -> float:
    dx = max(float(box[0] - point[0]), 0.0, float(point[0] - box[2]))
    dy = max(float(box[1] - point[1]), 0.0, float(point[1] - box[3]))
    return math.hypot(dx, dy)


def window_stats(
    tracks: dict[str, Any],
    table_config: dict[str, Any],
    staff_id: str,
    customer_id: str,
    start_frame: int,
    end_frame: int,
    *,
    samples: int = 12,
) -> WindowStats | None:
    fps = float(tracks["fps"])
    frames = np.rint(np.linspace(start_frame, end_frame, samples)).astype(np.float32)
    staff_frames, staff_observed = track_boxes(tracks, staff_id)
    customer_frames, customer_observed = track_boxes(tracks, customer_id)
    staff, staff_visible = interpolate_boxes(staff_frames, staff_observed, frames, max_gap=fps * 0.5)
    customer, customer_visible = interpolate_boxes(
        customer_frames, customer_observed, frames, max_gap=fps * 0.5
    )
    visible = staff_visible & customer_visible
    if visible.mean() < 0.5:
        return None
    staff = staff[visible]
    customer = customer[visible]
    staff_centers = (staff[:, :2] + staff[:, 2:]) / 2
    customer_centers = (customer[:, :2] + customer[:, 2:]) / 2
    person_scale = np.maximum(
        np.median(np.concatenate([staff[:, 3] - staff[:, 1], customer[:, 3] - customer[:, 1]])),
        1.0,
    )
    distances = np.linalg.norm(staff_centers - customer_centers, axis=1) / person_scale
    duration = max((end_frame - start_frame) / fps, 1e-6)
    staff_path = np.linalg.norm(np.diff(staff_centers, axis=0), axis=1).sum() / person_scale
    staff_speed = float(staff_path / duration)
    raw_tables = [np.asarray(row["box"], dtype=np.float32) for row in table_config.get("tables", [])]
    if raw_tables:
        table_distances = np.asarray(
            [[_point_box_distance(center, box) / person_scale for box in raw_tables] for center in staff_centers]
        )
        best_table = int(np.argmin(table_distances.mean(axis=0)))
        near_table_ratio = float((table_distances[:, best_table] < 0.75).mean())
        selected_table = tuple(float(value) for value in raw_tables[best_table])
    else:
        near_table_ratio = 0.0
        selected_table = None
    return WindowStats(
        staff_id=staff_id,
        customer_id=customer_id,
        start_frame=start_frame,
        end_frame=end_frame,
        mean_distance=float(distances.mean()),
        min_distance=float(distances.min()),
        staff_speed=staff_speed,
        near_table_ratio=near_table_ratio,
        table_box=selected_table,
    )


def _negative_scores(stats: WindowStats) -> dict[str, float]:
    proximity = max(0.0, 1.0 - stats.min_distance / 3.0)
    stillness = max(0.0, 1.0 - stats.staff_speed / 1.2)
    separation = min(1.0, stats.mean_distance / 3.0)
    return {
        "passing": proximity + min(1.0, stats.staff_speed / 1.2),
        "waiting": proximity + stillness,
        "cleanup": stats.near_table_ratio + separation + 0.5 * stillness,
    }


def _overlap_fraction(start: int, end: int, positive: dict[str, Any], fps: float) -> float:
    positive_start = float(positive["start"]) * fps
    positive_end = float(positive["end"]) * fps
    intersection = max(0.0, min(end, positive_end) - max(start, positive_start))
    return intersection / max(end - start, 1)


def bootstrap_annotations(
    flowmetric_root: Path,
    output: Path,
    *,
    tracks_relative: Path = Path("data/output/service_verifier_tables_run/tracks.json"),
    max_weak_per_class: int = 6,
    window_seconds: float = 2.5,
    stride_seconds: float = 1.0,
) -> dict[str, Any]:
    """Создать обучающие и проверочные аннотации из эталонных и слабых меток."""
    root = flowmetric_root.resolve()
    video_candidates = sorted((root / "data").glob("hf_*.mp4"))
    if len(video_candidates) != 1:
        raise ValueError(
            f"ожидался ровно один файл data/hf_*.mp4, найдено: {len(video_candidates)}"
        )
    video = video_candidates[0]
    tracks_path = root / tracks_relative
    gt_path = root / "benchmark" / "ground_truth.json"
    tables_path = root / "config" / "tables.json"
    tracks = load_json(tracks_path)
    gt = load_json(gt_path)
    tables = load_json(tables_path)
    mapping = map_gt_to_tracks(gt, tracks)
    fps = float(tracks["fps"])
    positives = service_intervals(gt)
    rows: list[dict[str, Any]] = []
    positive_pairs: set[tuple[str, str]] = set()
    for index, interval in enumerate(positives):
        staff_id = mapping.get(int(interval["staff"]))
        customer_id = mapping.get(int(interval["customer"]))
        if staff_id is None or customer_id is None:
            continue
        start_frame = max(0, int(round(float(interval["start"]) * fps)))
        end_frame = min(int(tracks["n_frames"]) - 1, int(round(float(interval["end"]) * fps)))
        _, customer_observed = track_boxes(tracks, customer_id)
        table = _table_box(tables, customer_observed)
        positive_pairs.add((staff_id, customer_id))
        rows.append(
            {
                "episode_id": f"gt_service_{index:03d}_{start_frame:06d}_{end_frame:06d}",
                "label": "service",
                "split": "train",
                "video": str(video),
                "tracks": str(tracks_path),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "staff_id": staff_id,
                "customer_id": customer_id,
                "table_box": table,
                "label_source": "ground_truth",
                "ground_truth_ids": {"staff": interval["staff"], "customer": interval["customer"]},
                "review_required": False,
            }
        )
    if len(rows) < 2:
        raise ValueError(
            "требуются как минимум два сопоставленных эталонных интервала обслуживания"
        )
    rows[-1]["split"] = "val"

    staff_ids = [mapping[person] for person in gt.get("staff", []) if person in mapping]
    customer_ids = [
        mapping[person]
        for person in sorted(ground_truth_people(gt) - set(int(p) for p in gt.get("staff", [])))
        if person in mapping
    ]
    window = max(2, int(round(window_seconds * fps)))
    stride = max(1, int(round(stride_seconds * fps)))
    candidates: list[WindowStats] = []
    for start in range(0, max(1, int(tracks["n_frames"]) - window), stride):
        end = min(int(tracks["n_frames"]) - 1, start + window)
        for staff_id in staff_ids:
            for customer_id in customer_ids:
                if (staff_id, customer_id) in positive_pairs and any(
                    mapping.get(int(interval["staff"])) == staff_id
                    and mapping.get(int(interval["customer"])) == customer_id
                    and _overlap_fraction(start, end, interval, fps) > 0.2
                    for interval in positives
                ):
                    continue
                stats = window_stats(tracks, tables, staff_id, customer_id, start, end)
                if stats is not None:
                    candidates.append(stats)

    selected_keys: set[tuple[str, str, int, int]] = set()
    for label in ("passing", "waiting", "cleanup"):
        ranked = sorted(candidates, key=lambda item: _negative_scores(item)[label], reverse=True)
        chosen: list[WindowStats] = []
        for stats in ranked:
            key = (stats.staff_id, stats.customer_id, stats.start_frame, stats.end_frame)
            if key in selected_keys:
                continue
            if any(
                stats.staff_id == previous.staff_id
                and abs(stats.start_frame - previous.start_frame) < window
                for previous in chosen
            ):
                continue
            chosen.append(stats)
            selected_keys.add(key)
            if len(chosen) == max_weak_per_class:
                break
        if len(chosen) < 2:
            raise ValueError(
                f"не удалось сформировать достаточно кандидатов со слабой меткой {label!r}"
            )
        for index, stats in enumerate(chosen):
            rows.append(
                {
                    "episode_id": f"weak_{label}_{index:03d}_{stats.start_frame:06d}_{stats.end_frame:06d}",
                    "label": label,
                    "split": "val" if index == len(chosen) - 1 else "train",
                    "video": str(video),
                    "tracks": str(tracks_path),
                    "start_frame": stats.start_frame,
                    "end_frame": stats.end_frame,
                    "staff_id": stats.staff_id,
                    "customer_id": stats.customer_id,
                    "table_box": stats.table_box,
                    "label_source": "weak_geometry_heuristic",
                    "review_required": True,
                    "weak_label_scores": _negative_scores(stats),
                    "window_stats": {
                        "mean_distance": stats.mean_distance,
                        "min_distance": stats.min_distance,
                        "staff_speed": stats.staff_speed,
                        "near_table_ratio": stats.near_table_ratio,
                    },
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "annotations": str(output),
        "mapping": {str(key): value for key, value in sorted(mapping.items())},
        "counts": {
            label: sum(row["label"] == label for row in rows)
            for label in ("service", "passing", "waiting", "cleanup")
        },
        "weak_labels_require_review": True,
    }
