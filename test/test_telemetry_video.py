"""Saved-array mapping/refusal tests. No renderer, simulator or controller runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import render_telemetry_video as renderer


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def tape(self, links=16, *, states=None, actions=None, times=None, raw=None):
        if states is None:
            states = np.zeros((1501, 2 * (links + 1)), dtype=np.float64)
        if actions is None:
            actions = np.zeros(1500, dtype=np.float32)
        if times is None:
            times = np.arange(1501, dtype=np.float64) / 50
        values = dict(states=states, actions=actions, times=times)
        if raw is not None:
            values["raw_actions"] = raw
        path = self.root / "tape.npz"
        np.savez(path, **values)
        return path

    def test_post_state_action_alignment_and_promote_force(self):
        states = np.zeros((1501, 34), dtype=np.float64)
        states[:, 0] = np.linspace(-.1, .1, 1501)
        actions = np.zeros(1500, dtype=np.float32)
        actions[0], actions[849], actions[850] = .1, -.2, .3
        saved = renderer.load_recording(self.tape(states=states, actions=actions), 16)
        rows = renderer.telemetry(saved, 850)
        self.assertEqual(rows[0]["cart_m"], states[1, 0])
        self.assertEqual(rows[0]["delivered_force_N"], float(actions[0]) * 80)
        self.assertNotEqual(rows[0]["delivered_force_N"], 8.)
        self.assertEqual(rows[0]["encoded_pts_seconds"], 0.)
        self.assertEqual(rows[0]["display_time_seconds"], .02)
        self.assertEqual(rows[849]["phase"], "HANGING SETTLE")
        self.assertEqual(rows[850]["phase"], "SWING UP AND BALANCE")
        self.assertEqual(rows[850]["delivered_force_N"], float(actions[850]) * 80)
        self.assertEqual(rows[-1]["post_command"], 1500)
        self.assertEqual(rows[-1]["encoded_pts_seconds"], 29.98)
        self.assertEqual(rows[-1]["display_time_seconds"], 30.)

    def test_initial_sample_excluded_strict_threshold_and_hold_latch(self):
        states = np.zeros((1501, 34), dtype=np.float64)
        states[1:1251, 1] = np.pi
        states[1250, 1] = .15  # Equality is outside the strict gate.
        saved = renderer.load_recording(self.tape(states=states), 16)
        rows = renderer.telemetry(saved, 850)
        self.assertEqual(rows[0]["upright_streak_ticks"], 0)
        self.assertEqual(rows[1249]["upright_streak_ticks"], 0)
        self.assertFalse(rows[1498]["hold_met"])
        self.assertEqual(rows[-1]["best_upright_ticks"], 250)
        self.assertTrue(rows[-1]["hold_met"])
        states[:] = 0.
        states[1:1250, 1] = np.pi
        states[1500, 1] = np.pi
        reset_rows = renderer.telemetry(renderer.load_recording(self.tape(states=states), 16), 850)
        self.assertEqual(reset_rows[-1]["upright_streak_ticks"], 0)
        self.assertEqual(reset_rows[-1]["best_upright_ticks"], 250)
        self.assertTrue(reset_rows[-1]["hold_met"])

    def test_relative_angles_are_accumulated_and_small_errors_preserved(self):
        states = np.zeros((1501, 6), dtype=np.float64)
        states[1, 1], states[1, 2] = .2, -.2
        states[2, 1] = 2 ** -55
        angle = renderer.absolute_angles(states, 2)
        np.testing.assert_array_equal(angle[1], [.2, 0.])
        self.assertEqual(angle[2, 0], 2 ** -55)
        recording = renderer.Recording(states, np.zeros(1500, dtype=np.float32), None, 2)
        x, z, length = renderer.geometry(recording)
        self.assertEqual(length, 1.5)
        self.assertAlmostEqual(x[1, 2], 1.5 * np.sin(.2))
        self.assertAlmostEqual(z[1, 2], .07 + 1.5 * np.cos(.2) + 1.5)

    def test_count_time_delivery_and_nonfinite_refusals(self):
        p = self.tape()
        with self.assertRaisesRegex(ValueError, "declared count"):
            renderer.load_recording(p, 15)
        with self.assertRaisesRegex(ValueError, "native clock"):
            renderer.load_recording(self.tape(times=np.arange(1501, dtype=np.float64) / 50 + .02), 16)
        actions = np.zeros(1500, dtype=np.float64)
        actions[0] = .1  # This binary64 value is not an actual F32 delivery.
        with self.assertRaisesRegex(ValueError, "float32 deliveries"):
            renderer.load_recording(self.tape(actions=actions), 16)
        states = np.zeros((1501, 34), dtype=np.float64)
        states[-1, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite complete"):
            renderer.load_recording(self.tape(states=states), 16)
        with self.assertRaisesRegex(ValueError, "aligned binary64 raw"):
            renderer.load_recording(self.tape(raw=np.zeros(1499, dtype=np.float64)), 16)

    def test_source_hash_ownership_and_literal_phase_refusals(self):
        p = self.tape()
        recording = renderer.load_recording(p, 16)
        row = renderer.file_row(p)
        protocol = dict(n_links=16, total_length_m=3., total_mass_kg=1., cart_mass_kg=1.,
                        physics_dt_s=.005, frame_skip=4, episode_seconds=30., force_limit_N=80.,
                        rail_limit_m=3., upright_limit_rad=.15, required_hold_seconds=5.,
                        cart_damping=.02, total_hinge_damping=.015, joint_armature=.0005,
                        capsule_radius_m=.025, reset_noise_std=.05)
        source = dict(stage="reserved", recording=row, seed=216500, exact=False,
                      source_pins_unchanged=True, reset_count_after_initialization=0,
                      identity=dict(protocol=protocol, policy_parameters=dict(park_ticks=850)),
                      score=dict(success=True, complete=True, ticks=1500,
                                 longest_upright_ticks=1500, longest_upright_seconds=30., maximum_cart_m=0.))
        receipt = self.root / "receipt.json"
        receipt.write_text(json.dumps(source))
        rows, layout = renderer.source_binding(receipt, row, recording, seed=216500, park_ticks=850)
        self.assertEqual(len(rows), 1500)
        self.assertEqual(layout, "original episode receipt")
        with self.assertRaisesRegex(ValueError, "phase boundary"):
            renderer.source_binding(receipt, row, recording, seed=216500, park_ticks=800)
        with self.assertRaisesRegex(ValueError, "not bound"):
            renderer.source_binding(receipt, {**row, "sha256": "0" * 64}, recording, seed=216500, park_ticks=850)
        source["identity"]["protocol"]["force_limit_N"] = 10
        receipt.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "different recorded task"):
            renderer.source_binding(receipt, row, recording, seed=216500, park_ticks=850)
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            renderer.checked_row(receipt, "0" * 64)


if __name__ == "__main__":
    unittest.main()
