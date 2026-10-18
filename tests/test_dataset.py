import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flowmetric_temporal.dataset import EpisodeDataset
from flowmetric_temporal.schema import TRAJECTORY_FEATURES


class DatasetTests(unittest.TestCase):
    def test_dataset_loads_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            time = 4
            crops = np.zeros((time, 16, 16, 3), dtype=np.uint8)
            np.savez_compressed(
                root / "e1.npz",
                staff=crops,
                customer=crops,
                table=crops,
                trajectory=np.zeros((time, len(TRAJECTORY_FEATURES)), dtype=np.float32),
                mask=np.ones(time, dtype=np.bool_),
            )
            row = {"episode_id": "e1", "label": "service", "split": "train", "npz": "e1.npz"}
            (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            item = EpisodeDataset(root / "manifest.jsonl", split="train")[0]
            self.assertEqual(item["staff"].shape, (time, 3, 16, 16))
            self.assertEqual(item["trajectory"].shape, (time, len(TRAJECTORY_FEATURES)))
            self.assertEqual(item["label"].item(), 0)
