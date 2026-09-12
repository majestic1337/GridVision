import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def repo_tmp_path():
    root = Path(__file__).resolve().parents[1]
    tmp_dir = root / "data" / "test_tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
