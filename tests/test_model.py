import unittest

import torch

from flowmetric_temporal.model import EpisodeClassifier, ModelConfig
from flowmetric_temporal.schema import EPISODE_LABELS, TRAJECTORY_FEATURES


class ModelTests(unittest.TestCase):
    def test_model_forward_and_backward(self) -> None:
        config = ModelConfig(
            visual_dim=8,
            trajectory_dim=8,
            hidden_dim=32,
            heads=4,
            layers=1,
            dropout=0.0,
            max_frames=8,
        )
        model = EpisodeClassifier(config)
        batch, time = 2, 4
        crops = torch.rand(batch, time, 3, 32, 32)
        trajectory = torch.rand(batch, time, len(TRAJECTORY_FEATURES))
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        logits = model(crops, crops, crops, trajectory, mask)
        self.assertEqual(logits.shape, (batch, len(EPISODE_LABELS)))
        logits.sum().backward()
        self.assertIsNotNone(model.head[-1].weight.grad)
