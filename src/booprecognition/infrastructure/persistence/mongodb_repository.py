"""Adaptador MongoDB: pessoas + embeddings com busca vetorial (KNN).

Estrutura:
    {persons}    { _id: personId, identification, created_at, updated_at, embedding_ids }
    {embeddings} { _id: embeddingId, embedding_id, person_id, vector: [512 floats], created_at }

Modos de busca:
    atlas        -> `$vectorSearch` (Atlas ou Community 8.2+ com mongot), cosseno
    brute_force  -> cosseno calculado em Python sobre todos os vetores (dev/testes, poucos dados)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from pymongo import ASCENDING
from pymongo.client_session import ClientSession
from pymongo.database import Database

from booprecognition.domain.entities import FaceEmbedding, Person
from booprecognition.domain.value_objects import Embedding, Neighbor

SearchMode = Literal["atlas", "brute_force"]


class MongoFaceRepository:
    """Implementa PersonRepository e EmbeddingIndex sobre o MongoDB."""

    def __init__(
        self,
        database: Database[dict[str, Any]],
        persons_collection: str = "persons",
        embeddings_collection: str = "embeddings",
        search_mode: SearchMode = "atlas",
        index_name: str = "boop_vector_index",
        num_candidates: int = 100,
        use_transactions: bool = True,
    ) -> None:
        self._db = database
        self._persons = database[persons_collection]
        self._embeddings = database[embeddings_collection]
        self._search_mode = search_mode
        self._index_name = index_name
        self._num_candidates = num_candidates
        self._use_transactions = use_transactions

    def ensure_indexes(self, dimension: int = 512) -> None:
        """Cria o índice por pessoa e, no modo atlas, o índice de busca vetorial."""
        self._embeddings.create_index([("person_id", ASCENDING)])
        if self._search_mode != "atlas":
            return
        from pymongo.operations import SearchIndexModel

        existing = {idx["name"] for idx in self._embeddings.list_search_indexes()}
        if self._index_name in existing:
            return
        self._embeddings.create_search_index(
            SearchIndexModel(
                name=self._index_name,
                type="vectorSearch",
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector",
                            "numDimensions": dimension,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "person_id"},
                    ]
                },
            )
        )

    # PersonRepository
    def get(self, person_id: int) -> Person | None:
        doc = self._persons.find_one({"_id": person_id})
        if doc is None:
            return None
        return Person(
            identification=int(doc["identification"]),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            embedding_ids=[int(e) for e in doc.get("embedding_ids", [])],
        )

    def exists(self, person_id: int) -> bool:
        return self._persons.count_documents({"_id": person_id}, limit=1) > 0

    def create(self, person: Person, embedding: FaceEmbedding) -> None:
        def write(session: ClientSession | None) -> None:
            self._persons.insert_one(self._person_doc(person), session=session)
            self._embeddings.insert_one(self._embedding_doc(embedding), session=session)

        self._atomic(write)

    def append_embedding(
        self, person: Person, embedding: FaceEmbedding, evicted_ids: Sequence[int]
    ) -> None:
        def write(session: ClientSession | None) -> None:
            self._embeddings.insert_one(self._embedding_doc(embedding), session=session)
            if evicted_ids:
                self._embeddings.delete_many({"_id": {"$in": list(evicted_ids)}}, session=session)
            self._persons.replace_one(
                {"_id": person.identification}, self._person_doc(person), session=session
            )

        self._atomic(write)

    def delete(self, person_id: int) -> None:
        def write(session: ClientSession | None) -> None:
            self._embeddings.delete_many({"person_id": person_id}, session=session)
            self._persons.delete_one({"_id": person_id}, session=session)

        self._atomic(write)

    # EmbeddingIndex
    def nearest(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        if self._search_mode == "brute_force":
            return self._nearest_brute_force(vector, k)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": "vector",
                    "queryVector": list(vector.values),
                    "numCandidates": max(self._num_candidates, k),
                    "limit": k,
                }
            },
            {
                "$project": {
                    "embedding_id": 1,
                    "person_id": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        # Atlas normaliza cosseno como score = (1 + cos) / 2
        return [
            Neighbor(int(d["embedding_id"]), int(d["person_id"]), 2.0 * float(d["score"]) - 1.0)
            for d in self._embeddings.aggregate(pipeline)
        ]

    def _nearest_brute_force(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        projection = {"embedding_id": 1, "person_id": 1, "vector": 1}
        scored = [
            Neighbor(
                int(d["embedding_id"]),
                int(d["person_id"]),
                vector.cosine_similarity(Embedding.from_iterable(d["vector"])),
            )
            for d in self._embeddings.find({}, projection)
        ]
        return sorted(scored, key=lambda n: -n.similarity)[:k]

    # helpers
    def _atomic(self, write: Callable[[ClientSession | None], None]) -> None:
        if not self._use_transactions:
            write(None)
            return
        with self._session() as session:
            session.with_transaction(lambda s: write(s))

    @contextmanager
    def _session(self) -> Iterator[ClientSession]:
        with self._db.client.start_session() as session:
            yield session

    @staticmethod
    def _person_doc(person: Person) -> dict[str, Any]:
        return {
            "_id": person.identification,
            "identification": person.identification,
            "created_at": person.created_at,
            "updated_at": person.updated_at,
            "embedding_ids": list(person.embedding_ids),
        }

    @staticmethod
    def _embedding_doc(embedding: FaceEmbedding) -> dict[str, Any]:
        return {
            "_id": embedding.embedding_id,
            "embedding_id": embedding.embedding_id,
            "person_id": embedding.person_id,
            "vector": list(embedding.vector.values),
            "created_at": embedding.created_at,
        }
