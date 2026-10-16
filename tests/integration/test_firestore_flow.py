"""Fluxo ponta a ponta com InsightFace + Firestore reais.

Requer:
  BOOP_TEST_FACE_A1, BOOP_TEST_FACE_A2 (mesma pessoa), BOOP_TEST_FACE_B (outra pessoa)
  credenciais do Firebase (GOOGLE_APPLICATION_CREDENTIALS) ou FIRESTORE_EMULATOR_HOST.
Rodar: pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from booprecognition.config.container import build_container
from booprecognition.config.settings import Settings

pytestmark = pytest.mark.integration


def _image(env: str) -> bytes:
    path = os.environ.get(env)
    if not path:
        pytest.skip(f"{env} não definido")
    return Path(path).read_bytes()


def test_register_identify_remove() -> None:
    a1, a2, b = _image("BOOP_TEST_FACE_A1"), _image("BOOP_TEST_FACE_A2"), _image("BOOP_TEST_FACE_B")
    services = build_container(
        Settings(persons_collection="persons_it", embeddings_collection="embeddings_it")
    )

    created = services.register_person.execute(a1)
    person_id = created.person.identification
    try:
        assert services.register_person.execute(a2).person.identification == person_id

        found = services.identify_person.execute(a2)
        assert found is not None and found.identification == person_id

        other = services.identify_person.execute(b)
        assert other is None or other.identification != person_id
    finally:
        services.remove_person.execute(person_id)

    assert services.identify_person.execute(a1) is None
