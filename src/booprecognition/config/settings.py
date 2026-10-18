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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_ENV = "BOOP_CONFIG"
TARGET_ENV = "BOOP_TARGET"
DEFAULT_CONFIG_PATH = Path("config.yaml")

EMBEDDING_PART_DIMENSION = 512  # tamanho do vetor de cada modelo ArcFace

Target = Literal["firestore", "mongodb", "sqlite", "memory"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CollectionsConfig(_Model):
    persons: str = "persons"
    embeddings: str = "embeddings"


class AlgorithmConfig(_Model):
    knn_k: int = Field(default=5, ge=1, le=1000)
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Vizinhos abaixo disso são considerados outra pessoa e não votam no KNN
    min_neighbor_similarity: float = Field(default=0.45, ge=0.0, le=1.0)
    # O vizinho mais próximo do vencedor precisa ter ao menos essa similaridade
    min_top_similarity: float = Field(default=0.6, ge=0.0, le=1.0)
    # Diferença mínima entre o vencedor e a segunda pessoa mais parecida
    min_margin: float = Field(default=0.1, ge=0.0, le=1.0)
    # Vizinhos onde a segunda pessoa é procurada; precisa passar do histórico
    # máximo de uma pessoa para que ela não esconda um sósia
    margin_search_k: int = Field(default=20, ge=1, le=1000)
    # Identificação só realimenta o histórico com rostos muito parecidos, para
    # que um falso positivo não "contamine" a pessoa com vetores de outra
    update_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Cadastro: similaridade com um rosto existente a partir da qual é a mesma pessoa
    duplicate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_embeddings_person: int = Field(default=10, ge=1)
    # Modelos InsightFace: o detector SCRFD-10G vem do primeiro; cada modelo
    # contribui com um bloco de 512 dimensões (ensemble = 512 × N)
    encoder_models: list[str] = Field(
        default_factory=lambda: ["buffalo_l", "antelopev2"], min_length=1
    )
    # Ensemble: cada modelo sozinho precisa ter ao menos esse cosseno com o vencedor
    min_model_agreement: float = Field(default=0.5, ge=0.0, le=1.0)
    min_det_score: float = Field(default=0.7, ge=0.0, le=1.0)  # rostos borrados/artificiais
    max_yaw_ratio: float = Field(default=0.45, ge=0.0)  # rosto muito de perfil
    id_digits: int = Field(default=9, ge=6, le=18)

    @model_validator(mode="after")
    def _margin_sees_past_one_person(self) -> AlgorithmConfig:
        if self.min_margin > 0 and self.margin_search_k <= self.max_embeddings_person:
            raise ValueError(
                "margin_search_k deve ser maior que max_embeddings_person "
                f"({self.max_embeddings_person}); senão o histórico de uma pessoa "
                "esconde a segunda mais parecida e min_margin não tem efeito"
            )
        return self

    @property
    def embedding_dimension(self) -> int:
        return EMBEDDING_PART_DIMENSION * len(self.encoder_models)


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
    dimension: int = Field(default=1024, ge=1)  # 512 × modelos; tabela vec0 do sqlite-vec


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
    # 0 = token sem expiração (sem claim `exp`)
    access_token_ttl_seconds: int = Field(default=3600, ge=0, le=86400)
    clients: list[OAuthClientConfig] = Field(default_factory=list)

    @field_validator("access_token_ttl_seconds")
    @classmethod
    def _ttl_zero_or_min(cls, value: int) -> int:
        if 0 < value < 60:
            raise ValueError("access_token_ttl_seconds deve ser 0 (sem expiração) ou >= 60")
        return value


class Settings(_Model):
    target: Target = "firestore"
    project: str = "booprecognition"
    setup_alg: AlgorithmConfig = Field(default_factory=AlgorithmConfig, alias="setup-alg")
    databases: DatabasesConfig = DatabasesConfig()
    auth: AuthConfig = AuthConfig()

    @model_validator(mode="after")
    def _sqlite_dimension_matches_models(self) -> Settings:
        expected = self.setup_alg.embedding_dimension
        if self.target == "sqlite" and self.databases.sqlite.vector_search.dimension != expected:
            raise ValueError(
                f"sqlite.vector_search.dimension deve ser {expected} "
                f"(512 × {len(self.setup_alg.encoder_models)} modelos em encoder_models)"
            )
        return self


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
