"""Test-only configuration fixtures; deployment continues to require its declared built overlay."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENGINE_FIELDS = ("engine_overlay_root", "engine_manifest", "engine_manifest_sha256", "engine_flags")


def example(*, overlay=None, tier=False):
    value = json.loads((ROOT / "config.example.json").read_text())
    if not tier:
        # Transport/engine tests do not enable the optional disk tier implicitly.
        value.pop("tier", None)
    elif overlay is None:
        raise ValueError("tier fixtures require a declared bound engine overlay")
    if overlay is None:
        # Legacy transport tests deliberately exercise the supported no-overlay profile.
        for field in ENGINE_FIELDS:
            value.pop(field, None)
    else:
        if not all(field in overlay for field in ENGINE_FIELDS):
            raise ValueError("overlay fixture must bind all four engine fields")
        value.update(overlay)
    return value


@contextmanager
def config_file(*, overlay=None, tier=False):
    # Parsing the actual serialized copy still exercises the real config implementation.
    with tempfile.TemporaryDirectory(prefix="prefill-config-fixture-") as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(example(overlay=overlay, tier=tier)))
        yield path


def prepared_example(mode=None, *, overlay=None, tier=False):
    from experiment import prepare_config
    with config_file(overlay=overlay, tier=tier) as path:
        return prepare_config(path, mode)


def loaded_example(*, overlay=None, tier=False):
    from common import load_config
    with config_file(overlay=overlay, tier=tier) as path:
        return load_config(path)
