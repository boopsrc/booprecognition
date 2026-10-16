"""Composition root: único lugar que conhece as implementações concretas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from booprecognition.application.ports import EmbeddingIndex, FaceEncoder, PersonRepository
from booprecognition.application.use_cases.face_matcher import FaceMatcher
from booprecognition.application.use_cases.identify_person import IdentifyPerson
from booprecognition.application.use_cases.register_person import RegisterPerson
from booprecognition.application.use_cases.remove_person import RemovePerson
from booprecognition.config.settings import Settings
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
    settings = settings or Settings()
    encoder = encoder or _build_encoder(settings)
    repo = repository or _build_repository(settings)
    persons: PersonRepository = repo
    index: EmbeddingIndex = repo

    ids, clock = RandomIdGenerator(settings.id_digits), SystemClock()
    matcher = FaceMatcher(
        encoder, index, KnnClassifier(k=settings.knn_k, threshold=settings.match_threshold)
    )
    return Container(
        identify_person=IdentifyPerson(
            matcher, persons, ids, clock, settings.max_embeddings_per_person
        ),
        register_person=RegisterPerson(matcher, persons, ids, clock),
        remove_person=RemovePerson(persons),
    )


def _build_encoder(settings: Settings) -> FaceEncoder:
    from booprecognition.infrastructure.encoders.insightface_encoder import InsightFaceEncoder

    return InsightFaceEncoder(model_name=settings.encoder_model)


def _build_repository(settings: Settings) -> Any:
    if settings.storage == "memory":
        from booprecognition.infrastructure.persistence.in_memory_repository import (
            InMemoryFaceRepository,
        )

        return InMemoryFaceRepository()

    from google.cloud import firestore

    from booprecognition.infrastructure.persistence.firestore_repository import (
        FirestoreFaceRepository,
    )

    return FirestoreFaceRepository(
        firestore.Client(project=settings.firestore_project),
        settings.persons_collection,
        settings.embeddings_collection,
    )
