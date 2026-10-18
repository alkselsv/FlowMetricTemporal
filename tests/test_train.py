import unittest

from flowmetric_temporal.train import classification_metrics


class TrainTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
        self.assertEqual(metrics, {"accuracy": 1.0, "macro_f1": 1.0})

    def test_classification_metrics_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([], [])
