"""File manifest shared by canonical freeze and execution authority guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .canonical_runtime_closure import runtime_binding_paths

SCHEMA='rage_mill.canonical_runtime_closure.v1'
COVERAGE_FIELDS=('python_packages_complete','package_native_libraries_complete',
                 'model_generated_libraries_complete','stdlib_complete')


def create_runtime_manifest(**collector_options):
    files=[]
    for path in runtime_binding_paths(**collector_options):
        data=Path(path).read_bytes()
        files.append(dict(path=path,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    packages=tuple(collector_options.get('package_names',('numpy','scipy','casadi','mujoco','mpmath')))
    models=tuple(str(Path(path).resolve(strict=True))for path in collector_options.get('model_directories',()))
    coverage=dict(python_packages_complete=True,package_native_libraries_complete=True,
                  model_generated_libraries_complete=True,
                  stdlib_complete=bool(collector_options.get('include_stdlib',True)))
    return dict(schema=SCHEMA,files=files,coverage=coverage,
                collection_scope=f'Complete selected packages {packages!r}, their native sibling libraries, retained model directories {models!r}, loaded runtime and declared extra files; actual postexecution loaded-file coverage must also pass.')


def validate_runtime_manifest(manifest,*,require_complete=True):
    """Rehash every canonical file row; refuse aliases, duplicates or mutation."""
    if (not isinstance(manifest,dict)or manifest.get('schema')!=SCHEMA
            or not isinstance(manifest.get('collection_scope'),str)
            or not isinstance(manifest.get('files'),list)or not manifest['files']):
        raise ValueError('canonical runtime closure manifest required')
    coverage=manifest.get('coverage')
    if (not isinstance(coverage,dict)or set(coverage)!=set(COVERAGE_FIELDS)
            or not all(type(coverage[name])is bool for name in COVERAGE_FIELDS)
            or require_complete and not all(coverage.values())):
        raise ValueError('runtime coverage declaration incomplete/invalid')
    result={}
    for row in manifest['files']:
        if not isinstance(row,dict)or set(row)!=set(('path','sha256','bytes')):
            raise ValueError('runtime file row invalid')
        path,expected,size=row['path'],row['sha256'],row['bytes']
        if (not isinstance(path,str)or not Path(path).is_absolute()
                or str(Path(path).resolve(strict=True))!=path or not Path(path).is_file()
                or path in result or not isinstance(expected,str)or len(expected)!=64
                or any(char not in '0123456789abcdef'for char in expected)
                or type(size)is not int or size<0):
            raise ValueError('runtime path/hash/size duplicate, aliased or invalid')
        data=Path(path).read_bytes()
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError(f'runtime file changed: {path}')
        result[path]=expected
    return result


def write_runtime_manifest(path,**collector_options):
    """Create fresh JSON; caller must pin this JSON in the frozen identity."""
    path=Path(path)
    if path.exists():
        raise ValueError('fresh runtime manifest output required')
    manifest=create_runtime_manifest(**collector_options)
    path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    data=path.read_bytes()
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
