"""Serviços de domínio puros."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from booprecognition.domain.entities import RecognitionResult
from booprecognition.domain.value_objects import Neighbor


@dataclass(frozen=True, slots=True)
class KnnClassifier:
    """Votação KNN ponderada pela similaridade de cosseno.

    precision = média das similaridades do vencedor × fração dos K votos dele.
    Assim, um vencedor com poucos votos ou vizinhos distantes perde confiança.
    Vizinhos abaixo de `min_similarity` são de outras pessoas (no ArcFace,
    rostos diferentes ficam abaixo de ~0.3) e não entram na votação; sem esse
    corte, uma pessoa com poucos embeddings nunca atinge o limiar.

    Duas travas extras contra falso positivo (sósias, avatares 3D, personagens
    de jogo): o vizinho mais próximo do vencedor precisa ter similaridade
    >= `min_top_similarity`, e precisa superar o melhor vizinho de qualquer
    outra pessoa por pelo menos `min_margin` (resultado ambíguo = None).
    """

    k: int
    threshold: float
    min_similarity: float = 0.3
    min_top_similarity: float = 0.0
    min_margin: float = 0.0

    def classify(self, neighbors: Sequence[Neighbor]) -> RecognitionResult | None:
        nearest = sorted(neighbors, key=lambda n: -n.similarity)[: self.k]
        considered = [n for n in nearest if n.similarity >= self.min_similarity]
        if not considered:
            return None

        votes: dict[int, list[Neighbor]] = defaultdict(list)
        for neighbor in considered:
            votes[neighbor.person_id].append(neighbor)

        winner_id, winner_votes = max(
            votes.items(), key=lambda item: sum(n.similarity for n in item[1])
        )
        mean_similarity = sum(n.similarity for n in winner_votes) / len(winner_votes)
        vote_ratio = len(winner_votes) / len(considered)
        precision = max(0.0, min(1.0, mean_similarity * vote_ratio))

        if precision < self.threshold:
            return None

        top = max(n.similarity for n in winner_votes)
        if top < self.min_top_similarity:
            return None
        runner_up = max(
            (n.similarity for n in considered if n.person_id != winner_id), default=None
        )
        if runner_up is not None and top - runner_up < self.min_margin:
            return None
        return RecognitionResult(
            identification=winner_id,
            precision=round(precision, 4),
            history=tuple(n.embedding_id for n in winner_votes),
        )
