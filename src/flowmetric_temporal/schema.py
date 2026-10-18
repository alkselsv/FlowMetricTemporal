"""Stable labels and feature contract shared by extraction and inference."""

from __future__ import annotations

EPISODE_LABELS = ("service", "passing", "waiting", "cleanup")

TRAJECTORY_FEATURES = (
    "staff_cx",
    "staff_cy",
    "staff_w",
    "staff_h",
    "customer_cx",
    "customer_cy",
    "customer_w",
    "customer_h",
    "table_cx",
    "table_cy",
    "table_w",
    "table_h",
    "staff_to_customer_dx",
    "staff_to_customer_dy",
    "staff_to_customer_distance",
    "staff_vx",
    "staff_vy",
    "customer_vx",
    "customer_vy",
    "joint_visible",
)

