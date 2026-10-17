"""Objetos de transferência entre a aplicação e as interfaces de entrega."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from booprecognition.domain.entities import RecognitionResult


@dataclass(frozen=True, slots=True)
class RecognitionOutput:
    identification: int
    precision: float
    history: list[int]

    @classmethod
    def from_result(cls, result: RecognitionResult) -> RecognitionOutput:
        return cls(result.identification, result.precision, list(result.history))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identification": self.identification,
            "precision": self.precision,
            "history": self.history,
        }


@dataclass(frozen=True, slots=True)
class RegisterOutput:
    person: RecognitionOutput
    created: bool  # False quando a pessoa já existia

    def to_dict(self) -> dict[str, Any]:
        # `id` repete `identification` para quem só precisa persistir a referência
        return {"id": self.person.identification, **self.person.to_dict(), "created": self.created}


@dataclass(frozen=True, slots=True)
class RemoveOutput:
    identification: int
    removed_embeddings: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "identification": self.identification,
            "removed_embeddings": self.removed_embeddings,
        }
