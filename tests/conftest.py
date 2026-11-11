import json
import os
from pathlib import Path

import pytest

# the final access check raises instead of filtering + logging (docs/ACCESS_CONTROL.md)
os.environ.setdefault("JARVIS_ACCESS_MODE", "debug")

from jarvis_m.data.loaders import load_jarvis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "tests" / "fixtures" / "toy_team.json"
TOY_ENTERPRISE = ROOT / "tests" / "fixtures" / "toy_enterprise.json"
ACCESS = str(ROOT / "configs" / "access" / "default.yaml")


@pytest.fixture(scope="session")
def toy():
    return load_jarvis(str(TOY))


@pytest.fixture(scope="session")
def toy_enterprise():
    return load_jarvis(str(TOY_ENTERPRISE))


@pytest.fixture
def write_json(tmp_path):
    def _w(rel, obj, lines=False):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if lines:
            p.write_text("\n".join(json.dumps(o) for o in obj) + "\n", encoding="utf-8")
        else:
            p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    return _w
