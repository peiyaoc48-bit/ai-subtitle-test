from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES = Path(__file__).parent.parent / "data" / "samples"


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def sample_srt():
    return (SAMPLES / "sample_zh.srt").read_text(encoding="utf-8")


@pytest.fixture(scope="session", params=["sample_zh.srt", "hard_zh.srt"])
def any_srt(request):
    return (SAMPLES / request.param).read_text(encoding="utf-8")