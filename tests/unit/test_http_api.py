from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from booprecognition.config.container import build_container
from booprecognition.config.settings import Settings
from booprecognition.infrastructure.persistence.in_memory_repository import (
    InMemoryFaceRepository,
)
from booprecognition.interfaces.http.fastapi_app import create_app
from tests.unit.test_use_cases import FakeEncoder


@pytest.fixture
def client() -> TestClient:
    container = build_container(
        Settings(storage="memory"), encoder=FakeEncoder(), repository=InMemoryFaceRepository()
    )
    return TestClient(create_app(container))


def upload(data: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("face.jpg", data, "image/jpeg")}


def test_full_flow(client: TestClient) -> None:
    assert client.post("/identify", files=upload(b"A1")).json() is None

    created = client.post("/persons", files=upload(b"A1"))
    assert created.status_code == 201
    person_id = created.json()["identification"]

    again = client.post("/persons", files=upload(b"A22"))
    assert again.status_code == 200
    assert again.json()["identification"] == person_id

    found = client.post("/identify", files=upload(b"A333")).json()
    assert found["identification"] == person_id
    assert set(found) == {"identification", "precision", "history"}

    removed = client.delete(f"/persons/{person_id}")
    assert removed.status_code == 200
    assert client.delete(f"/persons/{person_id}").status_code == 404
