"""Geração de identificadores aleatórios e relógio do sistema."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime


class RandomIdGenerator:
    """Inteiros aleatórios criptograficamente seguros com N dígitos."""

    def __init__(self, digits: int = 9) -> None:
        self._low = 10 ** (digits - 1)
        self._span = 10**digits - self._low

    def new_id(self) -> int:
        return self._low + secrets.randbelow(self._span)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
