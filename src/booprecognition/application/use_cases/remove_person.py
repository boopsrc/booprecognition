"""Serviço 3: remover uma pessoa e todos os seus embeddings."""

from __future__ import annotations

from booprecognition.application.dto import RemoveOutput
from booprecognition.application.ports import PersonRepository
from booprecognition.domain.errors import PersonNotFoundError


class RemovePerson:
    def __init__(self, persons: PersonRepository) -> None:
        self._persons = persons

    def execute(self, person_id: int) -> RemoveOutput:
        person = self._persons.get(person_id)
        if person is None:
            raise PersonNotFoundError(person_id)
        self._persons.delete(person_id)
        return RemoveOutput(person_id, len(person.embedding_ids))
