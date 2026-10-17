"""Lógica de busca compartilhada pelos casos de uso."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from booprecognition.application.ports import EmbeddingIndex, FaceEncoder
from booprecognition.domain.entities import RecognitionResult
from booprecognition.domain.services import KnnClassifier
from booprecognition.domain.value_objects import Embedding, Neighbor


@dataclass(frozen=True, slots=True)
class MatchAttempt:
    embedding: Embedding
    result: RecognitionResult | None
    neighbors: Sequence[Neighbor] = ()

    @property
    def closest(self) -> Neighbor | None:
        return max(self.neighbors, key=lambda n: n.similarity, default=None)


class FaceMatcher:
    def __init__(
        self, encoder: FaceEncoder, index: EmbeddingIndex, classifier: KnnClassifier
    ) -> None:
        self._encoder = encoder
        self._index = index
        self._classifier = classifier

    def match(self, image: bytes) -> MatchAttempt:
        embedding = self._encoder.encode(image)
        neighbors = self._index.nearest(embedding, self._classifier.k)
        return MatchAttempt(embedding, self._classifier.classify(neighbors), neighbors)
