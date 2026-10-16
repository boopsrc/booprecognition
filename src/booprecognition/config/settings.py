"""Configuração via variáveis de ambiente (prefixo BOOP_) ou arquivo .env."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOOP_", env_file=".env", extra="ignore")

    storage: Literal["firestore", "memory"] = "firestore"
    firestore_project: str | None = None
    persons_collection: str = "persons"
    embeddings_collection: str = "embeddings"

    encoder_model: str = "buffalo_l"
    knn_k: int = Field(default=5, ge=1, le=1000)
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_embeddings_per_person: int = Field(default=10, ge=1)
    id_digits: int = Field(default=9, ge=6, le=18)
