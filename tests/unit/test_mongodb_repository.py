"""Casos de uso rodando sobre o adaptador MongoDB (mongomock, busca brute_force)."""

from __future__ import annotations

import mongomock
import pytest

from booprecognition.config.container import build_container
from booprecognition.config.settings import Settings
from booprecognition.infrastructure.persistence.mongodb_repository import MongoFaceRepository
from tests.unit.test_use_cases import FakeEncoder


@pytest.fixture
def db():  # type: ignore[no-untyped-def]
    return mongomock.MongoClient()["boop_test"]


@pytest.fixture
def repo(db) -> MongoFaceRepository:  # type: ignore[no-untyped-def]
    repository = MongoFaceRepository(
        db, "persons", "embeddings", search_mode="brute_force", use_transactions=False
    )
    repository.ensure_indexes()
    return repository


def test_register_identify_fifo_and_remove(db, repo) -> None:  # type: ignore[no-untyped-def]
    services = build_container(Settings(target="mongodb"), encoder=FakeEncoder(), repository=repo)

    created = services.register_person.execute(b"A")
    person_id = created.person.identification
    assert created.created is True
    assert services.register_person.execute(b"A1").created is False

    for i in range(12):
        found = services.identify_person.execute(b"A" + b"x" * i)
        assert found is not None and found.identification == person_id

    person = repo.get(person_id)
    assert person is not None and len(person.embedding_ids) == 10
    assert db["embeddings"].count_documents({"person_id": person_id}) == 10
    assert sorted(person.embedding_ids) == sorted(
        d["_id"] for d in db["embeddings"].find({"person_id": person_id})
    )

    assert services.identify_person.execute(b"B") is None

    assert services.remove_person.execute(person_id).removed_embeddings == 10
    assert db["persons"].count_documents({}) == 0
    assert db["embeddings"].count_documents({}) == 0


def test_atlas_score_is_converted_to_cosine(repo: MongoFaceRepository) -> None:
    class FakeCollection:
        def aggregate(self, pipeline):  # type: ignore[no-untyped-def]
            assert "$vectorSearch" in pipeline[0]
            return [{"embedding_id": 1, "person_id": 9, "score": 0.95}]

    repo._search_mode = "atlas"
    repo._embeddings = FakeCollection()  # type: ignore[assignment]
    from booprecognition.domain.value_objects import Embedding

    (neighbor,) = repo.nearest(Embedding.from_iterable([1.0, 0.0]), k=5)
    assert neighbor.person_id == 9
    assert neighbor.similarity == pytest.approx(0.9)
