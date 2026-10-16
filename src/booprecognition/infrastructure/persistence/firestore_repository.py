"""Adaptador Firestore: pessoas + embeddings com busca vetorial nativa (KNN).

Estrutura:
    persons/{personId}                         -> dados da pessoa + embedding_ids (FIFO)
    persons/{personId}/embeddings/{embeddingId} -> vetor (Vector 512) + metadados

A busca usa `find_nearest` no collection group `embeddings`, que exige um índice
vetorial (ver README).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from booprecognition.domain.entities import FaceEmbedding, Person
from booprecognition.domain.value_objects import Embedding, Neighbor

_DISTANCE_FIELD = "_distance"
_BATCH_LIMIT = 450  # Firestore aceita até 500 operações por batch


class FirestoreFaceRepository:
    """Implementa PersonRepository e EmbeddingIndex sobre o Firestore."""

    def __init__(
        self,
        client: firestore.Client,
        persons_collection: str = "persons",
        embeddings_collection: str = "embeddings",
    ) -> None:
        self._db = client
        self._persons = client.collection(persons_collection)
        self._embeddings_name = embeddings_collection

    # PersonRepository
    def get(self, person_id: int) -> Person | None:
        snapshot = self._person_ref(person_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        return Person(
            identification=int(data["identification"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            embedding_ids=[int(e) for e in data.get("embedding_ids", [])],
        )

    def exists(self, person_id: int) -> bool:
        return bool(self._person_ref(person_id).get().exists)

    def create(self, person: Person, embedding: FaceEmbedding) -> None:
        batch = self._db.batch()
        batch.create(self._person_ref(person.identification), self._person_doc(person))
        batch.set(self._embedding_ref(embedding), self._embedding_doc(embedding))
        batch.commit()

    def append_embedding(
        self, person: Person, embedding: FaceEmbedding, evicted_ids: Sequence[int]
    ) -> None:
        person_ref = self._person_ref(person.identification)
        batch = self._db.batch()
        batch.set(self._embedding_ref(embedding), self._embedding_doc(embedding))
        for eid in evicted_ids:
            batch.delete(person_ref.collection(self._embeddings_name).document(str(eid)))
        batch.set(person_ref, self._person_doc(person))
        batch.commit()

    def delete(self, person_id: int) -> None:
        person_ref = self._person_ref(person_id)
        embeddings = list(person_ref.collection(self._embeddings_name).list_documents())
        for start in range(0, len(embeddings), _BATCH_LIMIT):
            batch = self._db.batch()
            for ref in embeddings[start : start + _BATCH_LIMIT]:
                batch.delete(ref)
            batch.commit()
        person_ref.delete()

    # EmbeddingIndex
    def nearest(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        query = self._db.collection_group(self._embeddings_name).find_nearest(
            vector_field="vector",
            query_vector=Vector(list(vector.values)),
            distance_measure=DistanceMeasure.COSINE,
            limit=k,
            distance_result_field=_DISTANCE_FIELD,
        )
        neighbors = []
        for snapshot in query.get():
            data = snapshot.to_dict() or {}
            neighbors.append(
                Neighbor(
                    embedding_id=int(data["embedding_id"]),
                    person_id=int(data["person_id"]),
                    similarity=1.0 - float(data[_DISTANCE_FIELD]),
                )
            )
        return neighbors

    # helpers
    def _person_ref(self, person_id: int) -> Any:
        return self._persons.document(str(person_id))

    def _embedding_ref(self, embedding: FaceEmbedding) -> Any:
        return (
            self._person_ref(embedding.person_id)
            .collection(self._embeddings_name)
            .document(str(embedding.embedding_id))
        )

    @staticmethod
    def _person_doc(person: Person) -> dict[str, Any]:
        return {
            "identification": person.identification,
            "created_at": person.created_at,
            "updated_at": person.updated_at,
            "embedding_ids": list(person.embedding_ids),
        }

    @staticmethod
    def _embedding_doc(embedding: FaceEmbedding) -> dict[str, Any]:
        return {
            "embedding_id": embedding.embedding_id,
            "person_id": embedding.person_id,
            "vector": Vector(list(embedding.vector.values)),
            "created_at": embedding.created_at,
        }
