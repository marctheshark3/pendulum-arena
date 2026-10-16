"""Prospective runtime bindings for bounded canonical numerical stages.

Metadata only: enumerate/hash source, native libraries, stdlib and retained
model-generated JIT artifacts without constructing a model or calling math.
The consumer must also verify actual loaded-file coverage after execution.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import sysconfig


def _runtime_source(path):
    return path.suffix in ('.py','.casadi','.c','.h')or '.so' in path.name


def runtime_binding_paths(*,model_directories=(),package_names=('numpy','scipy','casadi','mujoco','mpmath'),
                          extra_paths=(),include_loaded=True,include_stdlib=True):
    paths={Path(__file__).resolve(),Path(sys.executable).resolve()}
    paths.update(Path(path).resolve(strict=True)for path in extra_paths)
    if include_loaded:
        for module in tuple(sys.modules.values()):
            path=getattr(module,'__file__',None)
            if path and Path(path).is_file():
                paths.add(Path(path).resolve())
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields=line.split()
            if len(fields)>=6 and fields[-1].startswith('/')and Path(fields[-1]).is_file():
                paths.add(Path(fields[-1]).resolve())
    if include_stdlib:
        root=Path(sysconfig.get_path('stdlib'))
        paths.update(path.resolve()for path in root.rglob('*')
                     if path.is_file()and 'site-packages'not in path.parts and _runtime_source(path))
    for name in package_names:
        if name not in ('numpy','scipy','casadi','mujoco','mpmath'):
            raise ValueError('only explicit canonical numerical runtime packages permitted')
        spec=importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f'required runtime package unavailable: {name}')
        for location in spec.submodule_search_locations:
            folder=Path(location)
            paths.update(path.resolve()for path in folder.rglob('*')if path.is_file()and _runtime_source(path))
            libraries=folder.parent/f'{name}.libs'
            if libraries.exists():
                paths.update(path.resolve()for path in libraries.rglob('*')if path.is_file()and '.so'in path.name)
    for location in model_directories:
        folder=Path(location).resolve(strict=True)
        if not folder.is_dir():
            raise ValueError('retained model directory required')
        paths.update(path.resolve()for path in folder.rglob('*')if path.is_file()and _runtime_source(path))
    if not all(path.is_file()for path in paths):
        raise ValueError('every runtime binding must name an actual file')
    return sorted(map(str,paths))


def runtime_source_hashes(**kwargs):
    return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in runtime_binding_paths(**kwargs)}
