"""Composition root: único lugar que conhece as implementações concretas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from booprecognition.application.ports import EmbeddingIndex, FaceEncoder, PersonRepository
from booprecognition.application.use_cases.face_matcher import FaceMatcher
from booprecognition.application.use_cases.identify_person import IdentifyPerson
from booprecognition.application.use_cases.register_person import RegisterPerson
from booprecognition.application.use_cases.remove_person import RemovePerson
from booprecognition.config.settings import Settings, load_settings, read_secret
from booprecognition.domain.services import KnnClassifier
from booprecognition.infrastructure.ids.random_id_generator import RandomIdGenerator, SystemClock


@dataclass(frozen=True, slots=True)
class Container:
    identify_person: IdentifyPerson
    register_person: RegisterPerson
    remove_person: RemovePerson


def build_container(
    settings: Settings | None = None,
    encoder: FaceEncoder | None = None,
    repository: Any | None = None,
) -> Container:
    """Monta os casos de uso. `encoder`/`repository` permitem injetar dublês."""
    settings = settings or load_settings()
    alg = settings.setup_alg
    encoder = encoder or _build_encoder(settings)
    repo = repository or build_repository(settings)
    persons: PersonRepository = repo
    index: EmbeddingIndex = repo

    ids, clock = RandomIdGenerator(alg.id_digits), SystemClock()
    matcher = FaceMatcher(encoder, index, KnnClassifier(
        k=alg.knn_k,
        threshold=alg.match_threshold,
        min_similarity=alg.min_neighbor_similarity,
        min_top_similarity=alg.min_top_similarity,
        min_margin=alg.min_margin,
    ))
    return Container(
        identify_person=IdentifyPerson(
            matcher, persons, ids, clock, alg.max_embeddings_person,
            update_threshold=alg.update_threshold,
        ),
        register_person=RegisterPerson(
            matcher, persons, ids, clock, duplicate_threshold=alg.duplicate_threshold
        ),
        remove_person=RemovePerson(persons),
    )


def build_repository(settings: Settings) -> Any:
    """Escolhe a implementação de banco a partir de `target` no YAML."""
    factories: dict[str, Callable[[Settings], Any]] = {
        "firestore": _build_firestore,
        "mongodb": _build_mongodb,
        "sqlite": _build_sqlite,
        "memory": _build_memory,
    }
    return factories[settings.target](settings)


def _build_encoder(settings: Settings) -> FaceEncoder:
    from booprecognition.infrastructure.encoders.insightface_encoder import InsightFaceEncoder

    alg = settings.setup_alg
    return InsightFaceEncoder(
        model_names=alg.encoder_models,
        min_det_score=alg.min_det_score,
        max_yaw_ratio=alg.max_yaw_ratio,
    )


def _build_memory(_: Settings) -> Any:
    from booprecognition.infrastructure.persistence.in_memory_repository import (
        InMemoryFaceRepository,
    )

    return InMemoryFaceRepository()


def _build_firestore(settings: Settings) -> Any:
    from google.cloud import firestore

    from booprecognition.infrastructure.persistence.firestore_repository import (
        FirestoreFaceRepository,
    )

    cfg = settings.databases.firestore
    credentials_path = read_secret(cfg.credentials_env)
    client = (
        firestore.Client.from_service_account_json(credentials_path, project=cfg.project_id)
        if credentials_path
        else firestore.Client(project=cfg.project_id)  # Application Default Credentials
    )
    return FirestoreFaceRepository(client, cfg.collections.persons, cfg.collections.embeddings)


def _build_mongodb(settings: Settings) -> Any:
    from pymongo import MongoClient

    from booprecognition.infrastructure.persistence.mongodb_repository import (
        MongoFaceRepository,
    )

    cfg = settings.databases.mongodb
    uri = read_secret(cfg.uri_env)
    if not uri:
        raise RuntimeError(f"Variável de ambiente {cfg.uri_env} com a URI do MongoDB não definida")

    client: MongoClient[dict[str, Any]] = MongoClient(uri, appname=settings.project)
    repo = MongoFaceRepository(
        client[cfg.database],
        persons_collection=cfg.collections.persons,
        embeddings_collection=cfg.collections.embeddings,
        search_mode=cfg.vector_search.mode,
        index_name=cfg.vector_search.index_name,
        num_candidates=cfg.vector_search.num_candidates,
        use_transactions=cfg.use_transactions,
    )
    repo.ensure_indexes()
    return repo


def _build_sqlite(settings: Settings) -> Any:
    from booprecognition.infrastructure.persistence.sqlite_repository import (
        SqliteFaceRepository,
    )

    cfg = settings.databases.sqlite
    return SqliteFaceRepository(
        read_secret(cfg.path_env) or cfg.path,
        persons_table=cfg.tables.persons,
        embeddings_table=cfg.tables.embeddings,
        search_mode=cfg.vector_search.mode,
        dimension=cfg.vector_search.dimension,
    )
