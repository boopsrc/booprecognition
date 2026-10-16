"""Adaptador SQLite: banco local em arquivo, sem servidor nem credenciais.

Tabelas:
    {persons}    identification PK, created_at, updated_at, embedding_ids (JSON, FIFO)
    {embeddings} embedding_id PK, person_id FK (ON DELETE CASCADE), vector BLOB float32

Modos de busca:
    brute_force -> matriz numpy em memória (recarregada após escritas) e cosseno vetorizado
    sqlite_vec  -> extensão sqlite-vec, tabela virtual vec0 com distância cosseno
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np

from booprecognition.domain.entities import FaceEmbedding, Person
from booprecognition.domain.value_objects import Embedding, Neighbor

SearchMode = Literal["brute_force", "sqlite_vec"]


class SqliteFaceRepository:
    """Implementa PersonRepository e EmbeddingIndex sobre um arquivo SQLite."""

    def __init__(
        self,
        path: str | Path,
        persons_table: str = "boop_persons",
        embeddings_table: str = "boop_embeddings_history",
        search_mode: SearchMode = "brute_force",
        dimension: int = 512,
    ) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = RLock()
        self._persons = f'"{persons_table}"'
        self._embeddings = f'"{embeddings_table}"'
        self._vec = f'"{embeddings_table}_vec"'
        self._index_name = f'"idx_{embeddings_table}_person"'
        self._search_mode = search_mode
        self._dimension = dimension
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        if search_mode == "sqlite_vec":
            self._load_sqlite_vec()
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {self._persons} (
                    identification INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    embedding_ids TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {self._embeddings} (
                    embedding_id INTEGER PRIMARY KEY,
                    person_id INTEGER NOT NULL
                        REFERENCES {self._persons}(identification) ON DELETE CASCADE,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS {self._index_name}
                    ON {self._embeddings}(person_id);
                """
            )
            if self._search_mode == "sqlite_vec":
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec} USING vec0("
                    f"embedding_id INTEGER PRIMARY KEY, "
                    f"vector float[{self._dimension}] distance_metric=cosine)"
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # PersonRepository
    def get(self, person_id: int) -> Person | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM {self._persons} WHERE identification = ?", (person_id,)
            ).fetchone()
        if row is None:
            return None
        return Person(
            identification=int(row["identification"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            embedding_ids=[int(e) for e in json.loads(row["embedding_ids"])],
        )

    def exists(self, person_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self._persons} WHERE identification = ?", (person_id,)
            ).fetchone()
        return row is not None

    def create(self, person: Person, embedding: FaceEmbedding) -> None:
        with self._transaction() as conn:
            conn.execute(
                f"INSERT INTO {self._persons} VALUES (?, ?, ?, ?)", self._person_row(person)
            )
            self._insert_embedding(conn, embedding)

    def append_embedding(
        self, person: Person, embedding: FaceEmbedding, evicted_ids: Sequence[int]
    ) -> None:
        with self._transaction() as conn:
            self._insert_embedding(conn, embedding)
            self._delete_embeddings(conn, evicted_ids)
            conn.execute(
                f"UPDATE {self._persons} SET created_at = ?, updated_at = ?, embedding_ids = ? "
                f"WHERE identification = ?",
                (*self._person_row(person)[1:], person.identification),
            )

    def delete(self, person_id: int) -> None:
        with self._transaction() as conn:
            if self._search_mode == "sqlite_vec":
                ids = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT embedding_id FROM {self._embeddings} WHERE person_id = ?",
                        (person_id,),
                    )
                ]
                self._delete_vec(conn, ids)
            # embeddings saem via ON DELETE CASCADE
            conn.execute(f"DELETE FROM {self._persons} WHERE identification = ?", (person_id,))

    # EmbeddingIndex
    def nearest(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        if self._search_mode == "sqlite_vec":
            return self._nearest_sqlite_vec(vector, k)
        return self._nearest_brute_force(vector, k)

    def _nearest_brute_force(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        ids, persons, matrix = self._load_cache()
        if not len(ids):
            return []
        query = np.asarray(vector.values, dtype=np.float32)
        similarities = matrix @ query  # vetores já normalizados: produto = cosseno
        top = np.argsort(-similarities)[:k]
        return [Neighbor(int(ids[i]), int(persons[i]), float(similarities[i])) for i in top]

    def _nearest_sqlite_vec(self, vector: Embedding, k: int) -> Sequence[Neighbor]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT v.embedding_id, e.person_id, v.distance "
                f"FROM {self._vec} v JOIN {self._embeddings} e USING (embedding_id) "
                f"WHERE v.vector MATCH ? AND k = ? ORDER BY v.distance",
                (self._to_blob(vector), k),
            ).fetchall()
        return [Neighbor(int(r[0]), int(r[1]), 1.0 - float(r[2])) for r in rows]

    # helpers
    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")
            self._cache = None  # invalida a matriz do brute_force

    def _load_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if self._cache is None:
                rows = self._conn.execute(
                    f"SELECT embedding_id, person_id, vector FROM {self._embeddings}"
                ).fetchall()
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                persons = np.array([r[1] for r in rows], dtype=np.int64)
                matrix = (
                    np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                    if rows
                    else np.empty((0, 0), dtype=np.float32)
                )
                self._cache = (ids, persons, matrix)
            return self._cache

    def _insert_embedding(self, conn: sqlite3.Connection, embedding: FaceEmbedding) -> None:
        blob = self._to_blob(embedding.vector)
        conn.execute(
            f"INSERT INTO {self._embeddings} VALUES (?, ?, ?, ?, ?)",
            (
                embedding.embedding_id,
                embedding.person_id,
                blob,
                embedding.vector.dimension,
                embedding.created_at.isoformat(),
            ),
        )
        if self._search_mode == "sqlite_vec":
            conn.execute(
                f"INSERT INTO {self._vec}(embedding_id, vector) VALUES (?, ?)",
                (embedding.embedding_id, blob),
            )

    def _delete_embeddings(self, conn: sqlite3.Connection, ids: Sequence[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM {self._embeddings} WHERE embedding_id IN ({marks})", list(ids))
        self._delete_vec(conn, ids)

    def _delete_vec(self, conn: sqlite3.Connection, ids: Sequence[int]) -> None:
        if self._search_mode != "sqlite_vec" or not ids:
            return
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM {self._vec} WHERE embedding_id IN ({marks})", list(ids))

    def _load_sqlite_vec(self) -> None:
        try:
            import sqlite_vec  # type: ignore[import-not-found]

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except (ImportError, AttributeError, sqlite3.OperationalError) as exc:
            raise RuntimeError(
                "vector_search.mode=sqlite_vec exige `pip install booprecognition[sqlite-vec]` "
                "e um Python com suporte a enable_load_extension. "
                "Use mode: brute_force como alternativa."
            ) from exc

    @staticmethod
    def _to_blob(vector: Embedding) -> bytes:
        return np.asarray(vector.values, dtype=np.float32).tobytes()

    @staticmethod
    def _person_row(person: Person) -> tuple[Any, ...]:
        return (
            person.identification,
            person.created_at.isoformat(),
            person.updated_at.isoformat(),
            json.dumps(person.embedding_ids),
        )
