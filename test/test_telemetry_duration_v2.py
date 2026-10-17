"""Synthetic saved-data checks only; no physics, rendering or qualification."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/render_telemetry_video_v2.py'
spec = importlib.util.spec_from_file_location('telemetry_duration_v2_test', path)
renderer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = renderer
spec.loader.exec_module(renderer)


class DurationBinding(unittest.TestCase):
    def check_duration(self, seconds, ticks=402):
        states = np.zeros((1501, 38), dtype=np.float64)
        states[:1099, 1] = np.pi
        states.setflags(write=False)
        actions = np.zeros(1500, dtype=np.float32)
        recording = renderer.Recording(states, actions, None, 18)
        row = dict(sha256='a' * 64, bytes=1)
        protocol = dict(n_links=18, total_length_m=3., total_mass_kg=1., cart_mass_kg=1.,
            physics_dt_s=.005, frame_skip=4, episode_seconds=30., force_limit_N=80.,
            rail_limit_m=3., upright_limit_rad=.15, required_hold_seconds=5.,
            cart_damping=.02, total_hinge_damping=.015, joint_armature=.0005,
            capsule_radius_m=.025, reset_noise_std=.05)
        receipt = dict(stage='reserved', recording=row, seed=218500, exact=False,
            source_pins_unchanged=True, reset_count_after_initialization=0,
            identity=dict(protocol=protocol, policy_parameters=dict(park_ticks=850)),
            score=dict(success=True, complete=True, ticks=1500, longest_upright_ticks=ticks,
                       longest_upright_seconds=seconds, maximum_cart_m=0.))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'synthetic-receipt.json'
            source.write_text(json.dumps(receipt))
            return renderer.source_binding(source, row, recording, seed=218500, park_ticks=850)

    def test_native_multiplication_with_402_strict_ticks(self):
        self.assertNotEqual(402 * .02, 402 / 50)
        rows, _ = self.check_duration(402 * .02)
        self.assertEqual(rows[-1]['best_upright_ticks'], 402)

    def test_legacy_division_with_same_exact_tick_count(self):
        self.assertEqual(self.check_duration(402 / 50)[0][-1]['best_upright_ticks'], 402)

    def test_rejects_different_duration(self):
        with self.assertRaisesRegex(ValueError, 'recomputed complete'):
            self.check_duration(8.05)

    def test_rejects_different_tick_count(self):
        with self.assertRaisesRegex(ValueError, 'recomputed complete'):
            self.check_duration(402 * .02, ticks=403)


if __name__ == '__main__':
    unittest.main()
