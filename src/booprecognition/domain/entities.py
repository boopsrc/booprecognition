"""Entidades do domínio."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from booprecognition.domain.value_objects import Embedding


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    embedding_id: int
    person_id: int
    vector: Embedding
    created_at: datetime


@dataclass(slots=True)
class Person:
    identification: int
    created_at: datetime
    updated_at: datetime
    embedding_ids: list[int] = field(default_factory=list)  # do mais antigo ao mais novo

    def add_embedding(self, embedding_id: int, max_embeddings: int, now: datetime) -> list[int]:
        """Adiciona um embedding ao histórico e devolve os IDs descartados (FIFO)."""
        self.embedding_ids.append(embedding_id)
        self.updated_at = now
        overflow = len(self.embedding_ids) - max_embeddings
        if overflow <= 0:
            return []
        evicted = self.embedding_ids[:overflow]
        self.embedding_ids = self.embedding_ids[overflow:]
        return evicted


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    identification: int
    precision: float
    history: tuple[int, ...]
