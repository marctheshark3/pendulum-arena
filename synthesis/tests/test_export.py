from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


class ExportBoundaries(unittest.TestCase):
    def test_manifest_detects_changed_numeric_bytes(self):
        spec = importlib.util.spec_from_file_location('public_regenerate_test', ROOT/'regenerate.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = b'original numeric checkpoint'
            (base/'checkpoint.npz').write_bytes(data)
            (base/'original-manifest.json').write_text(json.dumps(dict(files=[dict(
                path='checkpoint.npz', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())])))
            module.ROOT = base
            self.assertEqual(module.verify()['verified_files'], 1)
            (base/'checkpoint.npz').write_bytes(b'changed numeric checkpoint!')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                module.verify()

    def test_stale_reference_cannot_use_saved_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)/'different-reference.npz'
            reference.write_bytes((ROOT/'data/15/physical-reference.npz').read_bytes()+b'changed-container')
            result = subprocess.run([sys.executable, str(ROOT/'regenerate.py'), 'feedback',
                '--count', '15', '--reference', str(reference), '--out', str(Path(directory)/'output')],
                capture_output=True, text=True, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('derivative checkpoint belongs to a different reference', result.stderr)

    def test_diagnostic_does_not_accept_a_reserved_seed(self):
        result = subprocess.run([sys.executable, str(ROOT/'regenerate.py'), 'diagnostic',
            '--count', '16', '--controller', 'unused', '--out', 'unused', '--seed', '216500'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('unrecognized arguments', result.stderr)

    def test_export_cannot_activate_a_seventeen_experiment(self):
        result = subprocess.run([sys.executable, str(ROOT/'regenerate.py'), 'feedback',
            '--count', '17', '--out', 'unused'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_released_sixteen_is_the_literal_probe07_prefix(self):
        original = load(ROOT/'data/16/probe07-controller.npz')
        released = load(ROOT/'data/16/released-controller.npz')
        for key in ('actions', 'gains', 'offsets', 'native_defects'):
            self.assertEqual(released[key].tobytes(), original[key][:650].tobytes(), key)
        self.assertEqual(released['states'].tobytes(), original['states'][:651].tobytes())
        for key in ('park_gain', 'balance_gain', 'parking_target', 'terminal_root'):
            self.assertEqual(released[key].tobytes(), original[key].tobytes(), key)
        self.assertEqual(int(original['park_ticks']), 800)
        self.assertEqual(int(released['park_ticks']), 850)

    def test_fifteen_physical_reference_is_a_recorded_native_prefix(self):
        reference = load(ROOT/'data/15/physical-reference.npz')
        recording = load(ROOT/'data/15/first-native-pass.npz')
        self.assertEqual(len(recording['actions']), 1500)
        self.assertEqual(reference['states'].tobytes(), recording['states'][:701].tobytes())
        self.assertEqual(reference['actions'].tobytes(), recording['actions'][:700].tobytes())
        self.assertTrue(np.array_equal(reference['actions'], reference['actions'].astype(np.float32).astype(float)))

    def test_original_sixteen_source_matches_historical_owner_pins(self):
        bindings = json.loads((ROOT/'evidence/16/source-stage-bindings.json').read_text())
        self.assertTrue(bindings)
        for row in bindings:
            self.assertTrue(row['matches_original_owner_pin'], row['file'])
            self.assertEqual(hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest(), row['sha256'])


if __name__ == '__main__':
    unittest.main()
