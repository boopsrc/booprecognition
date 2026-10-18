from booprecognition.domain.services import KnnClassifier
from booprecognition.domain.value_objects import Neighbor


def n(eid: int, pid: int, sim: float) -> Neighbor:
    return Neighbor(embedding_id=eid, person_id=pid, similarity=sim)


def test_returns_none_without_neighbors() -> None:
    assert KnnClassifier(k=5, threshold=0.5).classify([]) is None


def test_majority_wins_and_history_lists_voters() -> None:
    result = KnnClassifier(k=5, threshold=0.5).classify(
        [n(1, 10, 0.9), n(2, 10, 0.85), n(3, 10, 0.8), n(4, 20, 0.6), n(5, 10, 0.75)]
    )
    assert result is not None
    assert result.identification == 10
    assert result.history == (1, 2, 3, 5)
    assert 0.6 < result.precision <= 1.0


def test_below_threshold_returns_none() -> None:
    assert KnnClassifier(k=5, threshold=0.5).classify([n(1, 10, 0.3), n(2, 20, 0.29)]) is None


def test_only_k_nearest_are_considered() -> None:
    result = KnnClassifier(k=1, threshold=0.5).classify([n(1, 10, 0.6), n(2, 20, 0.95)])
    assert result is not None and result.identification == 20 and result.history == (2,)


def test_single_embedding_person_not_diluted_by_strangers() -> None:
    # Pessoa recém-cadastrada (1 embedding) cercada de vizinhos de outras pessoas
    neighbors = [n(1, 10, 0.62), n(2, 20, 0.12), n(3, 30, 0.08), n(4, 40, 0.05), n(5, 50, 0.02)]
    result = KnnClassifier(k=5, threshold=0.5).classify(neighbors)
    assert result is not None and result.identification == 10


def test_register_duplicate_guard_uses_closest_neighbor() -> None:
    from booprecognition.application.use_cases.face_matcher import MatchAttempt

    attempt = MatchAttempt(None, None, [n(1, 10, 0.31), n(2, 20, 0.5)])  # type: ignore[arg-type]
    assert attempt.closest is not None and attempt.closest.person_id == 20


def test_weak_top_similarity_is_rejected() -> None:
    # Avatar parecido: vários votos medianos, nenhum vizinho realmente próximo
    neighbors = [n(1, 10, 0.52), n(2, 10, 0.51), n(3, 10, 0.5)]
    classifier = KnnClassifier(k=5, threshold=0.5, min_top_similarity=0.55)
    assert classifier.classify(neighbors) is None


def test_ambiguous_between_two_people_is_rejected() -> None:
    neighbors = [n(1, 10, 0.70), n(2, 20, 0.66)]
    classifier = KnnClassifier(k=5, threshold=0.3, min_margin=0.08)
    assert classifier.classify(neighbors) is None
    assert KnnClassifier(k=5, threshold=0.3, min_margin=0.02).classify(neighbors) is not None


class _FixedEncoder:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self._vector = vector

    def encode(self, image: bytes):  # type: ignore[no-untyped-def]
        from booprecognition.domain.value_objects import Embedding

        return Embedding.from_iterable(self._vector)


class _FixedIndex:
    def __init__(self, neighbors: list[Neighbor]) -> None:
        self._neighbors = neighbors

    def nearest(self, vector, k):  # type: ignore[no-untyped-def]
        return self._neighbors


def _two_model_match(stored: tuple[float, ...]):  # type: ignore[no-untyped-def]
    from booprecognition.application.use_cases.face_matcher import FaceMatcher
    from booprecognition.domain.value_objects import Embedding

    query = (1.0, 0.0, 1.0, 0.0)  # bloco do modelo 1 | bloco do modelo 2
    stored_vec = Embedding.from_iterable(stored)
    sim = Embedding.from_iterable(query).cosine_similarity(stored_vec)
    neighbor = Neighbor(1, 10, sim, stored_vec)
    matcher = FaceMatcher(
        _FixedEncoder(query), _FixedIndex([neighbor]),
        KnnClassifier(k=5, threshold=0.3), model_parts=2, min_model_agreement=0.5,
    )
    return matcher.match(b"x")


def test_ensemble_models_agree() -> None:
    assert _two_model_match((1.0, 0.1, 1.0, 0.1)).result is not None


def test_ensemble_rejects_when_one_model_disagrees() -> None:
    # Média dos cossenos ~0.6 passaria, mas o 2º modelo (cosseno ~0.2) discorda
    attempt = _two_model_match((1.0, 0.0, 0.2, 0.98))
    assert attempt.result is None


def test_margin_finds_lookalike_beyond_k() -> None:
    # Pessoa 10 tem histórico cheio e ocupa os K=5 primeiros; o sósia (20) vem logo depois
    neighbors = [n(i, 10, 0.72 - i * 0.001) for i in range(1, 6)] + [n(9, 20, 0.69)]
    assert KnnClassifier(k=5, threshold=0.5, min_margin=0.1).classify(neighbors) is not None
    wide = KnnClassifier(k=5, threshold=0.5, min_margin=0.1, margin_search_k=20)
    assert wide.search_k == 20
    assert wide.classify(neighbors) is None


def test_margin_search_k_must_exceed_history() -> None:
    import pytest
    from pydantic import ValidationError

    from booprecognition.config.settings import AlgorithmConfig

    with pytest.raises(ValidationError, match="margin_search_k"):
        AlgorithmConfig(margin_search_k=10, max_embeddings_person=10)
