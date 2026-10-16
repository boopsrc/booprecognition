from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from booprecognition.config.container import build_repository
from booprecognition.config.settings import load_settings
from booprecognition.infrastructure.persistence.in_memory_repository import (
    InMemoryFaceRepository,
)

PROJECT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_project_config_yaml_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOP_TARGET", raising=False)
    settings = load_settings(PROJECT_CONFIG)
    assert settings.target in {"firestore", "mongodb", "memory"}
    assert settings.setup_alg.max_embeddings_person == 10
    assert settings.databases.mongodb.uri_env == "BOOP_MONGODB_URI"


def test_reads_target_algorithm_and_collections(tmp_path: Path) -> None:
    settings = load_settings(
        write(
            tmp_path,
            """
target: mongodb
project: demo
setup-alg:
  knn_k: 7
  match_threshold: 0.6
databases:
  mongodb:
    database: faces
    collections:
      persons: p
      embeddings: e
    vector_search:
      mode: brute_force
""",
        )
    )
    assert settings.target == "mongodb"
    assert settings.setup_alg.knn_k == 7
    assert settings.setup_alg.max_embeddings_person == 10  # padrão
    assert settings.databases.mongodb.collections.embeddings == "e"
    assert settings.databases.mongodb.vector_search.mode == "brute_force"


def test_env_overrides_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOP_TARGET", "memory")
    settings = load_settings(write(tmp_path, "target: firestore\n"))
    assert settings.target == "memory"
    assert isinstance(build_repository(settings), InMemoryFaceRepository)


def test_rejects_unknown_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOP_TARGET", raising=False)
    with pytest.raises(ValidationError):
        load_settings(write(tmp_path, "target: postgres\n"))


def test_rejects_typos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOP_TARGET", raising=False)
    with pytest.raises(ValidationError):
        load_settings(write(tmp_path, "databases:\n  mondodb: {}\n"))


def test_missing_explicit_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_mongodb_requires_uri_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BOOP_TARGET", raising=False)
    monkeypatch.delenv("BOOP_MONGODB_URI", raising=False)
    with pytest.raises(RuntimeError, match="BOOP_MONGODB_URI"):
        build_repository(load_settings(write(tmp_path, "target: mongodb\n")))
