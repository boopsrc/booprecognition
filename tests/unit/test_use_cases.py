from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest

from booprecognition.application.use_cases.face_matcher import FaceMatcher
from booprecognition.application.use_cases.identify_person import IdentifyPerson
from booprecognition.application.use_cases.register_person import RegisterPerson
from booprecognition.application.use_cases.remove_person import RemovePerson
from booprecognition.domain.errors import PersonNotFoundError
from booprecognition.domain.services import KnnClassifier
from booprecognition.domain.value_objects import Embedding
from booprecognition.infrastructure.persistence.in_memory_repository import (
    InMemoryFaceRepository,
)

# "Imagens" fake: o byte inicial define a pessoa, o resto adiciona uma pequena variação.
FACES = {b"A": (1.0, 0.0, 0.0), b"B": (0.0, 1.0, 0.0), b"C": (0.0, 0.0, 1.0)}


class FakeEncoder:
    def encode(self, image: bytes) -> Embedding:
        base = FACES[image[:1]]
        noise = len(image) * 0.01
        block = (base[0] + noise, base[1] + noise, base[2] + noise)
        return Embedding.from_iterable(block * 2)  # dois blocos, como o ensemble real


class SequentialIds:
    def __init__(self) -> None:
        self._counter = itertools.count(100_000_001)

    def new_id(self) -> int:
        return next(self._counter)


class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def repo() -> InMemoryFaceRepository:
    return InMemoryFaceRepository()


@pytest.fixture
def services(repo: InMemoryFaceRepository) -> tuple[RegisterPerson, IdentifyPerson, RemovePerson]:
    matcher = FaceMatcher(FakeEncoder(), repo, KnnClassifier(k=5, threshold=0.5))
    ids, clock = SequentialIds(), FixedClock()
    return (
        RegisterPerson(matcher, repo, ids, clock),
        IdentifyPerson(matcher, repo, ids, clock, max_embeddings=10),
        RemovePerson(repo),
    )


def test_identify_unknown_returns_none(services) -> None:  # type: ignore[no-untyped-def]
    _, identify, _ = services
    assert identify.execute(b"A1") is None


def test_register_then_identify(services, repo) -> None:  # type: ignore[no-untyped-def]
    register, identify, _ = services
    created = register.execute(b"A1")
    assert created.created is True

    result = identify.execute(b"A22")
    assert result is not None
    assert result.identification == created.person.identification
    assert result.history == created.person.history
    assert 0.5 <= result.precision <= 1.0
    assert repo.embedding_count(created.person.identification) == 2


def test_register_existing_returns_same_person(services, repo) -> None:  # type: ignore[no-untyped-def]
    register, _, _ = services
    first = register.execute(b"A1")
    second = register.execute(b"A333")
    assert second.created is False
    assert second.person.identification == first.person.identification
    assert repo.embedding_count() == 1


def test_different_people_get_different_ids(services) -> None:  # type: ignore[no-untyped-def]
    register, identify, _ = services
    a = register.execute(b"A").person.identification
    b = register.execute(b"B").person.identification
    assert a != b
    assert identify.execute(b"C") is None


def test_history_is_capped_fifo(services, repo) -> None:  # type: ignore[no-untyped-def]
    register, identify, _ = services
    person_id = register.execute(b"A").person.identification
    first_embedding = repo.get(person_id).embedding_ids[0]  # type: ignore[union-attr]
    for i in range(15):
        identify.execute(b"A" + b"x" * i)

    person = repo.get(person_id)
    assert person is not None
    assert len(person.embedding_ids) == 10
    assert first_embedding not in person.embedding_ids
    assert repo.embedding_count(person_id) == 10


def test_remove_deletes_everything(services, repo) -> None:  # type: ignore[no-untyped-def]
    register, identify, remove = services
    person_id = register.execute(b"A").person.identification
    identify.execute(b"A1")

    output = remove.execute(person_id)
    assert output.removed_embeddings == 2
    assert repo.embedding_count() == 0
    assert identify.execute(b"A1") is None


def test_remove_unknown_raises(services) -> None:  # type: ignore[no-untyped-def]
    _, _, remove = services
    with pytest.raises(PersonNotFoundError):
        remove.execute(123)
