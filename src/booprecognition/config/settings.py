"""Configuração lida de um arquivo YAML (config.yaml ou BOOP_CONFIG).

Segredos nunca ficam no YAML: ele guarda apenas o nome das variáveis de ambiente
que contêm credenciais/URIs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

CONFIG_ENV = "BOOP_CONFIG"
TARGET_ENV = "BOOP_TARGET"
DEFAULT_CONFIG_PATH = Path("config.yaml")

Target = Literal["firestore", "mongodb", "sqlite", "memory"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CollectionsConfig(_Model):
    persons: str = "persons"
    embeddings: str = "embeddings"


class AlgorithmConfig(_Model):
    knn_k: int = Field(default=5, ge=1, le=1000)
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_embeddings_person: int = Field(default=10, ge=1)
    encoder_model: str = "buffalo_l"
    id_digits: int = Field(default=9, ge=6, le=18)


class FirestoreConfig(_Model):
    project_id: str | None = None
    credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS"
    collections: CollectionsConfig = CollectionsConfig()


class MongoVectorSearchConfig(_Model):
    mode: Literal["atlas", "brute_force"] = "atlas"
    index_name: str = "boop_vector_index"
    num_candidates: int = Field(default=100, ge=1)


class MongoConfig(_Model):
    uri_env: str = "BOOP_MONGODB_URI"
    database: str = "booprecognition"
    collections: CollectionsConfig = CollectionsConfig()
    use_transactions: bool = True
    vector_search: MongoVectorSearchConfig = MongoVectorSearchConfig()


class SqliteTablesConfig(_Model):
    persons: str = Field(default="boop_persons", pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")
    embeddings: str = Field(default="boop_embeddings_history", pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")


class SqliteVectorSearchConfig(_Model):
    mode: Literal["brute_force", "sqlite_vec"] = "brute_force"
    dimension: int = Field(default=512, ge=1)  # usado pela tabela vec0 do sqlite-vec


class SqliteConfig(_Model):
    path: str = "./data/booprecognition.db"
    path_env: str = "BOOP_SQLITE_PATH"  # sobrescreve `path` quando definida
    tables: SqliteTablesConfig = SqliteTablesConfig()
    vector_search: SqliteVectorSearchConfig = SqliteVectorSearchConfig()


class DatabasesConfig(_Model):
    firestore: FirestoreConfig = FirestoreConfig()
    mongodb: MongoConfig = MongoConfig()
    sqlite: SqliteConfig = SqliteConfig()


Scope = Literal["identify", "persons:write", "persons:delete"]
ALL_SCOPES: tuple[Scope, ...] = ("identify", "persons:write", "persons:delete")


class OAuthClientConfig(_Model):
    client_id: str = Field(min_length=1)
    secret_env: str  # nome da env com o client_secret
    scopes: list[Scope] = Field(default_factory=lambda: list(ALL_SCOPES))


class AuthConfig(_Model):
    enabled: bool = True
    secret_env: str = "BOOP_JWT_SECRET"  # chave de assinatura HS256 (>= 32 bytes)
    algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    issuer: str = "booprecognition"
    audience: str = "booprecognition-api"
    access_token_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    clients: list[OAuthClientConfig] = Field(default_factory=list)


class Settings(_Model):
    target: Target = "firestore"
    project: str = "booprecognition"
    setup_alg: AlgorithmConfig = Field(default_factory=AlgorithmConfig, alias="setup-alg")
    databases: DatabasesConfig = DatabasesConfig()
    auth: AuthConfig = AuthConfig()


def load_settings(path: str | Path | None = None) -> Settings:
    """Lê o YAML. `BOOP_TARGET` sobrescreve o `target` (útil em deploy/testes)."""
    load_dotenv(override=False)  # segredos de um .env local, sem sobrescrever o ambiente
    config_path = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    elif path is not None or CONFIG_ENV in os.environ:
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    if target := os.environ.get(TARGET_ENV):
        data["target"] = target
    return Settings.model_validate(data)


def read_secret(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    return value or None
