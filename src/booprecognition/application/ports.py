"""Portas (interfaces) que a infraestrutura deve implementar."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from booprecognition.domain.entities import FaceEmbedding, Person
from booprecognition.domain.value_objects import Embedding, Neighbor


class FaceEncoder(Protocol):
    def encode(self, image: bytes) -> Embedding:
        """Detecta o rosto principal e devolve seu embedding normalizado."""
        ...


class EmbeddingIndex(Protocol):
    def nearest(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        """Retorna os K embeddings mais próximos (cosseno), do mais ao menos similar."""
        ...


class PersonRepository(Protocol):
    def get(self, person_id: int) -> Person | None: ...

    def exists(self, person_id: int) -> bool: ...

    def create(self, person: Person, embedding: FaceEmbedding) -> None:
        """Persiste pessoa nova com seu primeiro embedding (atômico)."""
        ...

    def append_embedding(
        self, person: Person, embedding: FaceEmbedding, evicted_ids: Sequence[int]
    ) -> None:
        """Salva novo embedding, remove os descartados e atualiza a pessoa (atômico)."""
        ...

    def delete(self, person_id: int) -> None:
        """Remove a pessoa e todos os seus embeddings."""
        ...


class IdGenerator(Protocol):
    def new_id(self) -> int: ...


class Clock(Protocol):
    def now(self) -> datetime: ...
