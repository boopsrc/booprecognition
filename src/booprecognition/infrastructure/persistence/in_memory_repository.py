"""Repositório em memória (testes e desenvolvimento local)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from threading import RLock

from booprecognition.domain.entities import FaceEmbedding, Person
from booprecognition.domain.value_objects import Embedding, Neighbor


class InMemoryFaceRepository:
    """Implementa PersonRepository e EmbeddingIndex com busca KNN por força bruta."""

    def __init__(self) -> None:
        self._persons: dict[int, Person] = {}
        self._embeddings: dict[int, FaceEmbedding] = {}
        self._lock = RLock()

    # PersonRepository
    def get(self, person_id: int) -> Person | None:
        with self._lock:
            person = self._persons.get(person_id)
            return copy.deepcopy(person) if person else None

    def exists(self, person_id: int) -> bool:
        return person_id in self._persons

    def create(self, person: Person, embedding: FaceEmbedding) -> None:
        with self._lock:
            self._persons[person.identification] = copy.deepcopy(person)
            self._embeddings[embedding.embedding_id] = embedding

    def append_embedding(
        self, person: Person, embedding: FaceEmbedding, evicted_ids: Sequence[int]
    ) -> None:
        with self._lock:
            self._embeddings[embedding.embedding_id] = embedding
            for eid in evicted_ids:
                self._embeddings.pop(eid, None)
            self._persons[person.identification] = copy.deepcopy(person)

    def delete(self, person_id: int) -> None:
        with self._lock:
            self._persons.pop(person_id, None)
            owned = [e.embedding_id for e in self._embeddings.values() if e.person_id == person_id]
            for eid in owned:
                del self._embeddings[eid]

    # EmbeddingIndex
    def nearest(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        with self._lock:
            scored = [
                Neighbor(
                    e.embedding_id, e.person_id, vector.cosine_similarity(e.vector), e.vector
                )
                for e in self._embeddings.values()
            ]
        return sorted(scored, key=lambda n: -n.similarity)[:k]

    def embedding_count(self, person_id: int | None = None) -> int:
        return sum(
            1 for e in self._embeddings.values() if person_id is None or e.person_id == person_id
        )
