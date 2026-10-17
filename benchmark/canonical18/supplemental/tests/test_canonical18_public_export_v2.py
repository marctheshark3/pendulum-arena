"""Read-only source/copy regression; no scientific imports or qualification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/marctheshark/.t3/artifacts/rage-mill-canonical18-20261016')
OLD = ROOT / 'scripts/export_canonical18_public_evidence.py'
NEW = ROOT / 'scripts/export_canonical18_public_evidence_v2.py'
IDENTITY = BASE / 'native-value-candidate01/identity.json'
IDENTITY_SHA = '34aa4bafd3f50d577c54a13ce3c518185a287a2da40d14b549c8a2bda81cf220'


def module(path):
    spec = importlib.util.spec_from_file_location('metadata_export_' + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def source_loop(path):
    """Compile only the exact exporter identity-source copy loop, without assemble."""
    tree = ast.parse(path.read_text())
    assemble = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'assemble')
    loops = [node for node in assemble.body if isinstance(node, ast.For)
             and ast.unparse(node.iter) == "enumerate(identity['sources'])"]
    if len(loops) != 1:
        raise AssertionError('exact source copy loop must remain unique')
    loop = ast.Module(body=loops, type_ignores=[])
    return compile(ast.fix_missing_locations(loop), str(path), 'exec')


def copy_sources(path, rows, destination):
    code = module(path)
    copier = code.Copier(destination)
    context = dict(Path=Path, identity={'sources': rows}, checkout=ROOT,
                   source_copy=copier, mappings=set(), need=code.need)
    exec(source_loop(path), context)
    return copier


class SourceCopyV2Tests(unittest.TestCase):
    def test_only_one_dispatch_condition_changed(self):
        old = OLD.read_bytes()
        needle = b"elif path.name in ('model.xml','runtime-closure.json'):"
        self.assertEqual(old.count(needle), 1)
        self.assertEqual(NEW.read_bytes(), old.replace(needle, b"elif path.name == 'model.xml':"))

    def test_actual_closures_have_distinct_bytes_and_keep_original_identity(self):
        self.assertEqual(hashlib.sha256(IDENTITY.read_bytes()).hexdigest(), IDENTITY_SHA)
        identity = json.loads(IDENTITY.read_text())
        rows = [row for row in identity['sources'] if Path(row['path']).name == 'runtime-closure.json']
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['sha256'] for row in rows}), 2)
        with tempfile.TemporaryDirectory() as directory:
            copied = copy_sources(NEW, rows, Path(directory))
            self.assertEqual(len({item['relative_path'] for item in copied.mapping}), 2)
            for item in copied.mapping:
                self.assertTrue(item['relative_path'].startswith('provenance/original/'))
                self.assertIn(item['original']['sha256'][:12], item['relative_path'])
                self.assertEqual((Path(directory)/item['relative_path']).read_bytes(),
                                 Path(item['original']['path']).read_bytes())

    def test_original_dispatch_reproduces_actual_collision(self):
        rows = [row for row in json.loads(IDENTITY.read_text())['sources']
                if Path(row['path']).name == 'runtime-closure.json']
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'conflicting public copy'):
                copy_sources(OLD, rows, Path(directory))

    def test_full_actual_identity_source_inventory_has_no_conflict(self):
        rows = json.loads(IDENTITY.read_text())['sources']
        with tempfile.TemporaryDirectory() as directory:
            copied = copy_sources(NEW, rows, Path(directory))
            self.assertEqual(len(copied.mapping), len({row['path'] for row in rows}))
            destinations = {item['relative_path'] for item in copied.mapping}
            self.assertEqual(len(destinations), len(copied.mapping))
            self.assertIn('payload/model.xml', destinations)
            self.assertNotIn('payload/runtime-closure.json', destinations)
            for item in copied.mapping:
                self.assertEqual(hashlib.sha256((Path(directory)/item['relative_path']).read_bytes()).hexdigest(),
                                 item['original']['sha256'])

    def test_synthetic_same_basename_and_same_digest_remain_distinct(self):
        code = module(NEW)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = base/'first/runtime-closure.json'; second = base/'second/runtime-closure.json'
            model = base/'model.xml'
            for path in (first, second):
                path.parent.mkdir(parents=True); path.write_text('{"saved": true}\n')
            model.write_text('<mujoco/>\n')
            copied = copy_sources(NEW, [code.pin(first),code.pin(second),code.pin(model)],base/'output')
            paths = [item['relative_path'] for item in copied.mapping]
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(paths[0].startswith('provenance/original/000-'))
            self.assertTrue(paths[1].startswith('provenance/original/001-'))
            self.assertEqual(paths[2], 'payload/model.xml')


if __name__ == '__main__':
    unittest.main(verbosity=2)
