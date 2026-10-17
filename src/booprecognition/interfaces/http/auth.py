"""OAuth 2.0 (grant `client_credentials`, RFC 6749 §4.4) com access tokens JWT.

Os clientes ficam em `auth.clients` no YAML; o segredo de cada um e a chave de
assinatura vêm de variáveis de ambiente, nunca do arquivo.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import Depends, Form, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from booprecognition.config.settings import AuthConfig, read_secret

MIN_SECRET_BYTES = 32
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class OAuthClient:
    client_id: str
    secret: str
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class TokenClaims:
    client_id: str
    scopes: frozenset[str]


class OAuthError(Exception):
    """Erro no formato do RFC 6749 §5.2."""

    def __init__(self, error: str, description: str, status_code: int = 400) -> None:
        super().__init__(description)
        self.error, self.description, self.status_code = error, description, status_code


class TokenService:
    def __init__(self, config: AuthConfig, signing_key: str, clients: list[OAuthClient]) -> None:
        if len(signing_key.encode()) < MIN_SECRET_BYTES:
            raise RuntimeError(f"A chave JWT precisa ter pelo menos {MIN_SECRET_BYTES} bytes")
        self._config = config
        self._key = signing_key
        self._clients = {c.client_id: c for c in clients}
        # Com TTL 0 os tokens não têm `exp`; se houver `exp`, ele continua sendo validado.
        self._required_claims = ["iat", "sub", "aud", "iss"]
        if config.access_token_ttl_seconds:
            self._required_claims.append("exp")

    @classmethod
    def from_config(cls, config: AuthConfig) -> TokenService:
        key = read_secret(config.secret_env)
        if not key:
            raise RuntimeError(f"Variável de ambiente {config.secret_env} (chave JWT) não definida")
        clients = []
        for c in config.clients:
            secret = read_secret(c.secret_env)
            if not secret:
                raise RuntimeError(f"Variável de ambiente {c.secret_env} não definida")
            clients.append(OAuthClient(c.client_id, secret, frozenset(c.scopes)))
        return cls(config, key, clients)

    def issue(self, client_id: str, client_secret: str, scope: str | None) -> dict[str, Any]:
        client = self._clients.get(client_id)
        # compare_digest sempre roda, para não vazar por tempo se o client existe
        expected = client.secret if client else uuid.uuid4().hex
        if not hmac.compare_digest(expected.encode(), client_secret.encode()) or client is None:
            raise OAuthError("invalid_client", "Credenciais do cliente inválidas", 401)

        requested = frozenset(scope.split()) if scope else client.scopes
        if not requested <= client.scopes:
            raise OAuthError("invalid_scope", "Escopo não permitido para este cliente")

        now = int(time.time())
        ttl = self._config.access_token_ttl_seconds
        claims = {
            "iss": self._config.issuer,
            "aud": self._config.audience,
            "sub": client.client_id,
            "iat": now,
            "nbf": now,
            "jti": uuid.uuid4().hex,
            "scope": " ".join(sorted(requested)),
        }
        if ttl:
            claims["exp"] = now + ttl
        token = jwt.encode(claims, self._key, algorithm=self._config.algorithm)
        response = {"access_token": token, "token_type": "Bearer", "scope": claims["scope"]}
        if ttl:
            response["expires_in"] = ttl
        return response

    def verify(self, token: str) -> TokenClaims:
        claims = jwt.decode(
            token,
            self._key,
            algorithms=[self._config.algorithm],  # fixo: impede "alg: none"/troca de algoritmo
            audience=self._config.audience,
            issuer=self._config.issuer,
            options={"require": self._required_claims},
        )
        return TokenClaims(claims["sub"], frozenset(str(claims.get("scope", "")).split()))


def parse_basic(header: str | None) -> tuple[str, str] | None:
    """Credenciais via `Authorization: Basic` (forma recomendada pelo RFC 6749 §2.3.1)."""
    if not header or not header.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    client_id, sep, secret = decoded.partition(":")
    return (client_id, secret) if sep else None


def token_endpoint(service: TokenService) -> Callable[..., Any]:
    async def token(
        request: Request,
        grant_type: Annotated[str, Form()],
        scope: Annotated[str | None, Form()] = None,
        client_id: Annotated[str | None, Form()] = None,
        client_secret: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        if grant_type != "client_credentials":
            raise OAuthError("unsupported_grant_type", "Use grant_type=client_credentials")
        basic = parse_basic(request.headers.get("authorization"))
        cid, secret = basic or (client_id, client_secret)
        if not cid or secret is None:
            raise OAuthError("invalid_client", "Credenciais do cliente ausentes", 401)
        return service.issue(cid, secret, scope)

    return token


def scope_guard(service: TokenService | None) -> Callable[..., Any]:
    """Dependência que valida o Bearer JWT e os escopos exigidos pela rota."""
    async def guard(
        security_scopes: SecurityScopes,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> TokenClaims | None:
        if service is None:  # auth.enabled = false
            return None
        challenge = {"WWW-Authenticate": "Bearer"}
        if credentials is None:
            raise HTTPException(401, "Token de acesso ausente", headers=challenge)
        try:
            claims = service.verify(credentials.credentials)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                401, "Token expirado", headers={"WWW-Authenticate": 'Bearer error="invalid_token"'}
            ) from None
        except jwt.InvalidTokenError:
            raise HTTPException(
                401, "Token inválido", headers={"WWW-Authenticate": 'Bearer error="invalid_token"'}
            ) from None
        missing = set(security_scopes.scopes) - claims.scopes
        if missing:
            raise HTTPException(
                403,
                f"Escopo insuficiente: {' '.join(sorted(missing))}",
                headers={"WWW-Authenticate": f'Bearer error="insufficient_scope", '
                         f'scope="{security_scopes.scope_str}"'},
            )
        return claims

    return guard
