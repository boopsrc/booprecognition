"""Value objects imutáveis do domínio."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from booprecognition.domain.errors import InvalidEmbeddingError


@dataclass(frozen=True, slots=True)
class Embedding:
    """Vetor facial L2-normalizado."""

    values: tuple[float, ...]

    @classmethod
    def from_iterable(cls, raw: Iterable[float]) -> Embedding:
        values = tuple(float(v) for v in raw)
        norm = math.sqrt(sum(v * v for v in values))
        if not values or norm == 0.0:
            raise InvalidEmbeddingError("Embedding vazio ou nulo")
        return cls(tuple(v / norm for v in values))

    @property
    def dimension(self) -> int:
        return len(self.values)

    def cosine_similarity(self, other: Embedding) -> float:
        if self.dimension != other.dimension:
            raise InvalidEmbeddingError("Dimensões diferentes")
        return sum(a * b for a, b in zip(self.values, other.values, strict=True))


@dataclass(frozen=True, slots=True)
class Neighbor:
    """Vizinho retornado pela busca KNN."""

    embedding_id: int
    person_id: int
    similarity: float  # cosseno em [-1, 1]
