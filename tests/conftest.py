import shutil
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def repo_tmp_path():
    tmp_dir = ROOT / "data" / "test_tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
