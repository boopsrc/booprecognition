"""Serviço 2: cadastrar uma pessoa nova (ou devolver a existente)."""

from __future__ import annotations

from booprecognition.application.dto import RecognitionOutput, RegisterOutput
from booprecognition.application.ports import Clock, IdGenerator, PersonRepository
from booprecognition.application.use_cases.face_matcher import FaceMatcher
from booprecognition.domain.entities import FaceEmbedding, Person, RecognitionResult


class RegisterPerson:
    _MAX_ID_ATTEMPTS = 10

    def __init__(
        self,
        matcher: FaceMatcher,
        persons: PersonRepository,
        ids: IdGenerator,
        clock: Clock,
        duplicate_threshold: float = 0.45,
    ) -> None:
        self._duplicate_threshold = duplicate_threshold
        self._matcher = matcher
        self._persons = persons
        self._ids = ids
        self._clock = clock

    def execute(self, image: bytes) -> RegisterOutput:
        attempt = self._matcher.match(image)
        if attempt.result is not None and self._persons.exists(attempt.result.identification):
            return RegisterOutput(RecognitionOutput.from_result(attempt.result), created=False)

        # Trava contra duplicidade: mesmo que o KNN não feche o voto, um rosto
        # muito parecido com um já cadastrado é tratado como a mesma pessoa.
        closest = attempt.closest
        if (
            closest is not None
            and closest.similarity >= self._duplicate_threshold
            and self._persons.exists(closest.person_id)
        ):
            existing = RecognitionResult(
                closest.person_id, round(closest.similarity, 4), (closest.embedding_id,)
            )
            return RegisterOutput(RecognitionOutput.from_result(existing), created=False)

        now = self._clock.now()
        person_id = self._unique_person_id()
        embedding = FaceEmbedding(self._ids.new_id(), person_id, attempt.embedding, now)
        person = Person(person_id, now, now, [embedding.embedding_id])
        self._persons.create(person, embedding)
        return RegisterOutput(
            RecognitionOutput(person_id, 1.0, [embedding.embedding_id]), created=True
        )

    def _unique_person_id(self) -> int:
        for _ in range(self._MAX_ID_ATTEMPTS):
            candidate = self._ids.new_id()
            if not self._persons.exists(candidate):
                return candidate
        raise RuntimeError("Não foi possível gerar um identificador único")
