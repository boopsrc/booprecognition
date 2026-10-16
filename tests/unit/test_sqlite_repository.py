"""Casos de uso rodando sobre o adaptador SQLite (arquivo real em tmp_path)."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from booprecognition.config.container import Container, build_container, build_repository
from booprecognition.config.settings import Settings, load_settings
from booprecognition.infrastructure.persistence.sqlite_repository import SqliteFaceRepository
from tests.unit.test_use_cases import FakeEncoder


def services(repo: SqliteFaceRepository, knn_k: int = 5) -> Container:
    settings = Settings.model_validate({"target": "sqlite", "setup-alg": {"knn_k": knn_k}})
    return build_container(settings, encoder=FakeEncoder(), repository=repo)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "boop.db"


@pytest.fixture
def repo(db_path: Path) -> SqliteFaceRepository:
    repository = SqliteFaceRepository(db_path)
    yield repository  # type: ignore[misc]
    repository.close()


def count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def test_register_identify_fifo_and_cascade(repo: SqliteFaceRepository, db_path: Path) -> None:
    svc = services(repo)
    created = svc.register_person.execute(b"A")
    person_id = created.person.identification
    assert created.created is True
    assert svc.register_person.execute(b"A1").created is False

    for i in range(12):
        found = svc.identify_person.execute(b"A" + b"x" * i)
        assert found is not None and found.identification == person_id

    person = repo.get(person_id)
    assert person is not None and len(person.embedding_ids) == 10
    assert count(db_path, "boop_embeddings_history") == 10
    assert svc.identify_person.execute(b"B") is None

    assert svc.remove_person.execute(person_id).removed_embeddings == 10
    assert count(db_path, "boop_persons") == 0
    assert count(db_path, "boop_embeddings_history") == 0


def test_data_survives_reopen(db_path: Path) -> None:
    first = SqliteFaceRepository(db_path)
    person_id = services(first).register_person.execute(b"A").person.identification
    first.close()

    reopened = SqliteFaceRepository(db_path)
    try:
        found = services(reopened).identify_person.execute(b"A22")
        assert found is not None and found.identification == person_id
    finally:
        reopened.close()


def test_concurrent_identify(repo: SqliteFaceRepository) -> None:
    # knn_k=1 isola o teste do SQLite da penalidade de votos do KnnClassifier
    svc = services(repo, knn_k=1)
    a = svc.register_person.execute(b"A").person.identification
    b = svc.register_person.execute(b"B").person.identification

    images = [b"A" + b"x" * (i % 5) if i % 2 else b"B" + b"y" * (i % 5) for i in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(svc.identify_person.execute, images))

    assert all(r is not None for r in results)
    assert {r.identification for r in results if r} == {a, b}
    for pid in (a, b):
        person = repo.get(pid)
        assert person is not None and len(person.embedding_ids) == 10


def test_failed_transaction_rolls_back(repo: SqliteFaceRepository, db_path: Path) -> None:
    svc = services(repo)
    svc.register_person.execute(b"A")
    person = repo.get(next(iter(_ids(db_path))))
    assert person is not None

    with pytest.raises(sqlite3.IntegrityError):
        repo.create(person, _embedding_like(repo, person.identification))  # PK duplicada
    assert count(db_path, "boop_embeddings_history") == 1


def test_container_builds_sqlite_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOP_TARGET", raising=False)
    db_file = tmp_path / "from_env.db"
    monkeypatch.setenv("BOOP_SQLITE_PATH", str(db_file))
    config = tmp_path / "config.yaml"
    config.write_text("target: sqlite\n", encoding="utf-8")

    repository = build_repository(load_settings(config))
    try:
        assert isinstance(repository, SqliteFaceRepository)
        assert db_file.exists()
    finally:
        repository.close()


def test_sqlite_vec_mode_without_extension_has_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "sqlite_vec", None)
    with pytest.raises(RuntimeError, match="brute_force"):
        SqliteFaceRepository(tmp_path / "vec.db", search_mode="sqlite_vec")


def _ids(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute('SELECT identification FROM "boop_persons"')]


def _embedding_like(repo: SqliteFaceRepository, person_id: int):  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from booprecognition.domain.entities import FaceEmbedding
    from booprecognition.domain.value_objects import Embedding

    return FaceEmbedding(
        999, person_id, Embedding.from_iterable([1.0, 0.0, 0.0]), datetime.now(UTC)
    )
