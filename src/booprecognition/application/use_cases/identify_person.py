"""Serviço 1: identificar uma pessoa por imagem."""

from __future__ import annotations

from booprecognition.application.dto import RecognitionOutput
from booprecognition.application.ports import Clock, IdGenerator, PersonRepository
from booprecognition.application.use_cases.face_matcher import FaceMatcher
from booprecognition.domain.entities import FaceEmbedding


class IdentifyPerson:
    def __init__(
        self,
        matcher: FaceMatcher,
        persons: PersonRepository,
        ids: IdGenerator,
        clock: Clock,
        max_embeddings: int,
        update_threshold: float = 0.0,
    ) -> None:
        self._update_threshold = update_threshold
        self._matcher = matcher
        self._persons = persons
        self._ids = ids
        self._clock = clock
        self._max_embeddings = max_embeddings

    def execute(self, image: bytes) -> RecognitionOutput | None:
        attempt = self._matcher.match(image)
        if attempt.result is None:
            return None

        person = self._persons.get(attempt.result.identification)
        if person is None:  # índice desatualizado em relação às pessoas
            return None

        # Só aprende com rostos muito parecidos: um falso positivo aceito no
        # limite não pode virar referência e puxar outros rostos para a pessoa.
        top = max(
            (n.similarity for n in attempt.neighbors if n.person_id == person.identification),
            default=0.0,
        )
        if top < self._update_threshold:
            return RecognitionOutput.from_result(attempt.result)

        now = self._clock.now()
        embedding = FaceEmbedding(self._ids.new_id(), person.identification, attempt.embedding, now)
        evicted = person.add_embedding(embedding.embedding_id, self._max_embeddings, now)
        self._persons.append_embedding(person, embedding, evicted)
        return RecognitionOutput.from_result(attempt.result)
