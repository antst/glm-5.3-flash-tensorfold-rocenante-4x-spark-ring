"""Explicit immutable TensorFold source overlays for the isolated prefill study."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

FLAG_VALUES = {
    'TF_GLM_HC_EXCHANGE': {'gather', 'p2p'},
    'TF_GLM_PROMPT_OWN_FRONT': {'0', '1'},
    'TF_GLM_PROMPT_MICROBATCH': {'0', '1'},
    'TF_GLM_PREFILL_ROWS': {'2048', '4096'},
    'TF_GLM_SIDE': {'0', '1'},
    'TF_GLM_PROMPT_PARTIALS': {'fp32'},
    'TF_GLM_MSA': {'0'},
    'TF_GLM_ONEPASS_SMALL_WARPS': {'4','8'},
    'TF_GLM_HC_RING_RELAY': {'0','1'},
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def flags(config):
    values = config.get('engine_flags', {})
    if not isinstance(values, dict) or set(values) - set(FLAG_VALUES):
        raise ValueError('Unknown prefill study engine flags')
    if any(type(v) is not str or v not in FLAG_VALUES[k] for k, v in values.items()):
        raise ValueError('Unsupported prefill study engine flag value')
    return dict(values)

def manifest(path, expected_sha):
    path = Path(path)
    if not re.fullmatch('[0-9a-f]{64}', str(expected_sha)) or sha(path) != expected_sha:
        raise RuntimeError('Engine manifest digest changed')
    value = json.loads(path.read_text())
    if value.get('schema') != 'glm-prefill-engine-source/v1':
        raise ValueError('Wrong engine manifest schema')
    files = value.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty engine source manifest')
    for name, wanted in files.items():
        p = PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name
                or not re.fullmatch('[0-9a-f]{64}', str(wanted))):
            raise ValueError('Unsafe engine source manifest entry')
    return value

def verify_tree(root, value):
    root = Path(root).resolve(strict=True)
    actual = set()
    for path in root.rglob('*'):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise RuntimeError('Engine source symlink: ' + str(path))
        if path.is_file():
            name = str(path.relative_to(root))
            actual.add(name)
            if value['files'].get(name) != sha(path):
                raise RuntimeError('Engine source changed: ' + name)
    if actual != set(value['files']):
        raise RuntimeError('Missing engine sources')
    return dict(value['files'])

def identity(config, *, verify=False):
    if 'engine_overlay_root' not in config:
        return None
    root = Path(config['engine_overlay_root'])
    path = Path(config['engine_manifest'])
    if not root.is_absolute() or not path.is_absolute():
        raise ValueError('Engine overlay and manifest need absolute paths')
    value = manifest(path, config['engine_manifest_sha256'])
    if verify:
        verify_tree(root / 'tensorfold', value)
    return {'manifest_sha256': config['engine_manifest_sha256'],
            'source_id': value['source_id'], 'files': value['files'],
            'engine_flags': flags(config)}

def environment(config):
    result = flags(config)
    if identity(config) is not None:
        result.update(PYTHONPATH='/opt/glm-prefill/engine',
                      PYTHONDONTWRITEBYTECODE='1',
                      SPARK_RING_ENGINE_SOURCE_ROOT='/opt/glm-prefill/engine/tensorfold',
                      SPARK_RING_ENGINE_MANIFEST='/opt/glm-prefill/engine-manifest.json',
                      SPARK_RING_ENGINE_MANIFEST_SHA256=config['engine_manifest_sha256'])
    return result

def mounts(config):
    if identity(config) is None:
        return []
    return ['--mount', 'type=bind,src='+config['engine_overlay_root']+',dst=/opt/glm-prefill/engine,readonly',
            '--mount', 'type=bind,src='+config['engine_manifest']+',dst=/opt/glm-prefill/engine-manifest.json,readonly']

def runtime_verify(imported_root):
    path = os.environ.get('SPARK_RING_ENGINE_MANIFEST')
    if not path:
        return None
    value = manifest(path, os.environ['SPARK_RING_ENGINE_MANIFEST_SHA256'])
    expected_root = Path(os.environ['SPARK_RING_ENGINE_SOURCE_ROOT']).resolve(strict=True)
    if Path(imported_root).resolve(strict=True) != expected_root:
        raise RuntimeError('TensorFold imported from outside the declared overlay')
    return verify_tree(expected_root, value)
