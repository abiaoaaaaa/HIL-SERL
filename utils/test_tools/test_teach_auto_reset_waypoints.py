import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/experiments/marvin_usb_insertion/teach_auto_reset_waypoints.py"
)
SPEC = importlib.util.spec_from_file_location("teach_auto_reset_waypoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TeachAutoResetWaypointsTest(unittest.TestCase):
    def test_saved_record_wraps_return_angle_error(self):
        args = argparse.Namespace(
            robot_ip="test", arm="A", drop_range_mm=(20.0, 10.0, 0.0)
        )
        initial = np.array([1.0, 2.0, 3.0, 179.0, 0.0, 0.0])
        waypoints = {
            name: initial.copy()
            for name in (
                "initial_pose",
                "usb_above",
                "usb_grasp",
                "extract_retreat",
                "drop_center",
            )
        }
        return_pose = np.array([2.0, 2.0, 3.0, -179.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "waypoints.json"
            MODULE.save_waypoints(output, args, waypoints, return_pose)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(saved["schema_version"], 1)
        self.assertEqual(saved["drop_random_half_range_mm"], [20.0, 10.0, 0.0])
        self.assertEqual(saved["return_error_xyzabc"][0], 1.0)
        self.assertAlmostEqual(saved["return_error_xyzabc"][3], 2.0)


if __name__ == "__main__":
    unittest.main()
