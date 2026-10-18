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

    def cosine_by_part(self, other: Embedding, parts: int) -> tuple[float, ...]:
        """Cosseno de cada bloco do vetor (um bloco por modelo do ensemble)."""
        if self.dimension != other.dimension or self.dimension % parts:
            raise InvalidEmbeddingError("Dimensões incompatíveis com o número de modelos")
        size = self.dimension // parts
        result = []
        for start in range(0, self.dimension, size):
            a, b = self.values[start : start + size], other.values[start : start + size]
            norm = math.sqrt(sum(v * v for v in a)) * math.sqrt(sum(v * v for v in b))
            result.append(sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class Neighbor:
    """Vizinho retornado pela busca KNN."""

    embedding_id: int
    person_id: int
    similarity: float  # cosseno em [-1, 1]
    # Vetor armazenado; usado para conferir a concordância de cada modelo do ensemble
    vector: Embedding | None = None
