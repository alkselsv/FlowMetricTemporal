import unittest

import numpy as np

from flowmetric_temporal.features import interpolate_boxes, trajectory_features
from flowmetric_temporal.schema import TRAJECTORY_FEATURES


class FeatureTests(unittest.TestCase):
    def test_interpolation_and_visibility(self) -> None:
        observed_frames = np.asarray([0, 10], dtype=np.float32)
        observed_boxes = np.asarray([[0, 0, 20, 40], [10, 20, 30, 60]], dtype=np.float32)
        target = np.asarray([0, 5, 20], dtype=np.float32)
        boxes, visible = interpolate_boxes(observed_frames, observed_boxes, target, max_gap=5)
        np.testing.assert_allclose(boxes[1], [5, 10, 25, 50])
        self.assertEqual(visible.tolist(), [True, True, False])


    def test_trajectory_feature_contract(self) -> None:
        frames = np.asarray([0, 5, 10])
        staff = np.asarray([[0, 0, 20, 40], [5, 0, 25, 40], [10, 0, 30, 40]], dtype=np.float32)
        customer = np.asarray([[40, 0, 60, 40]] * 3, dtype=np.float32)
        result = trajectory_features(
            staff,
            customer,
            np.asarray([20, 20, 80, 80]),
            np.asarray([True, True, False]),
            width=100,
            height=100,
            fps=10,
            frames=frames,
        )
        self.assertEqual(result.shape, (3, len(TRAJECTORY_FEATURES)))
        np.testing.assert_allclose(result[:, -1], [1, 1, 0])
        self.assertGreater(result[1, TRAJECTORY_FEATURES.index("staff_vx")], 0)
