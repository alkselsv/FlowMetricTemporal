"""Temporal episode classification for FlowMetric."""

from .model import EpisodeClassifier, ModelConfig
from .schema import EPISODE_LABELS, TRAJECTORY_FEATURES

__all__ = ["EPISODE_LABELS", "TRAJECTORY_FEATURES", "EpisodeClassifier", "ModelConfig"]
__version__ = "0.1.0"

