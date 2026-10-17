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
