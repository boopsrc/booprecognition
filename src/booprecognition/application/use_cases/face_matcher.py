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
    """Encoda, busca os K vizinhos e decide.

    Com ensemble (`model_parts` > 1) o vetor é a concatenação de um bloco por
    modelo e a similaridade do índice é a média deles. Além da média, cada
    modelo precisa reconhecer a pessoa sozinho: o cosseno de cada bloco entre o
    rosto e o melhor vizinho do vencedor deve ser >= `min_model_agreement`.
    """

    def __init__(
        self,
        encoder: FaceEncoder,
        index: EmbeddingIndex,
        classifier: KnnClassifier,
        model_parts: int = 1,
        min_model_agreement: float = 0.0,
    ) -> None:
        self._encoder = encoder
        self._index = index
        self._classifier = classifier
        self._model_parts = model_parts
        self._min_model_agreement = min_model_agreement

    def match(self, image: bytes) -> MatchAttempt:
        embedding = self._encoder.encode(image)
        neighbors = self._index.nearest(embedding, self._classifier.search_k)
        result = self._classifier.classify(neighbors)
        if result is not None and not self._models_agree(embedding, result, neighbors):
            result = None
        return MatchAttempt(embedding, result, neighbors)

    def _models_agree(
        self, embedding: Embedding, result: RecognitionResult, neighbors: Sequence[Neighbor]
    ) -> bool:
        if self._model_parts <= 1:
            return True
        best = max(
            (n for n in neighbors if n.person_id == result.identification),
            key=lambda n: n.similarity,
        )
        if best.vector is None:  # índice sem vetor: não dá para conferir, recusa
            return False
        return all(
            sim >= self._min_model_agreement
            for sim in embedding.cosine_by_part(best.vector, self._model_parts)
        )
