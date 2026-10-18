import unittest

from flowmetric_temporal.bootstrap import map_gt_to_tracks, service_intervals


class BootstrapTests(unittest.TestCase):
    def test_maps_head_anchor_to_track(self) -> None:
        gt = {"anchors": {"note": "ignored", "0": {"7": [50, 20]}}}
        tracks = {
            "fps": 10,
            "tracks": {
                "42": {
                    "frames": [0, 10, 20],
                    "boxes": [[30, 10, 70, 110], [30, 10, 70, 110], [30, 10, 70, 110]],
                }
            },
        }
        self.assertEqual(map_gt_to_tracks(gt, tracks), {7: "42"})

    def test_pairs_service_boundaries_and_skips_zero_length(self) -> None:
        gt = {
            "duration_s": 10.0,
            "events": [
                {"t": 0.0, "type": "service_end", "staff": 1, "customer": 2},
                {"t": 2.0, "type": "service_start", "staff": 1, "customer": 3},
                {"t": 5.0, "type": "service_end", "staff": 1, "customer": 3},
                {"t": 8.0, "type": "service_start", "staff": 4, "customer": 5},
            ],
        }
        self.assertEqual(
            service_intervals(gt),
            [
                {"staff": 1, "customer": 3, "start": 2.0, "end": 5.0},
                {"staff": 4, "customer": 5, "start": 8.0, "end": 10.0},
            ],
        )

