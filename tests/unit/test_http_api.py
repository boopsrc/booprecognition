from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from booprecognition.config.container import build_container
from booprecognition.config.settings import AuthConfig, Settings
from booprecognition.infrastructure.persistence.in_memory_repository import (
    InMemoryFaceRepository,
)
from booprecognition.interfaces.http.auth import OAuthClient, TokenService
from booprecognition.interfaces.http.fastapi_app import create_app
from tests.unit.test_use_cases import FakeEncoder

SIGNING_KEY = "k" * 32
ALL = frozenset({"identify", "persons:write", "persons:delete"})


def make_tokens(ttl: int = 3600) -> TokenService:
    clients = [
        OAuthClient("admin", "s3cret", ALL),
        OAuthClient("reader", "r3ad", frozenset({"identify"})),
    ]
    return TokenService(AuthConfig(access_token_ttl_seconds=ttl), SIGNING_KEY, clients)


def make_app(tokens: TokenService) -> TestClient:
    container = build_container(
        Settings(target="memory"), encoder=FakeEncoder(), repository=InMemoryFaceRepository()
    )
    return TestClient(create_app(container, tokens))


def bearer(c: TestClient, client_id: str = "admin", secret: str = "s3cret") -> dict[str, str]:
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": secret}
    token = c.post("/oauth/token", data=data).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client() -> TestClient:
    c = make_app(make_tokens())
    c.headers.update(bearer(c))
    return c


def upload(data: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("face.jpg", data, "image/jpeg")}


def test_full_flow(client: TestClient) -> None:
    assert client.post("/identify", files=upload(b"A1")).json() is None

    created = client.post("/persons", files=upload(b"A1"))
    assert created.status_code == 201
    person_id = created.json()["identification"]
    assert created.json()["id"] == person_id
    assert created.headers["location"] == f"/persons/{person_id}"

    again = client.post("/persons", files=upload(b"A22"))
    assert again.status_code == 200
    assert again.json()["identification"] == person_id
    assert again.json()["id"] == person_id

    found = client.post("/identify", files=upload(b"A333")).json()
    assert found["identification"] == person_id
    assert set(found) == {"identification", "precision", "history"}

    removed = client.delete(f"/persons/{person_id}")
    assert removed.status_code == 200
    assert client.delete(f"/persons/{person_id}").status_code == 404


def test_token_via_basic_auth_and_response_shape() -> None:
    c = make_app(make_tokens())
    data = {"grant_type": "client_credentials"}
    resp = c.post("/oauth/token", data=data, auth=("admin", "s3cret"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "Bearer" and body["expires_in"] == 3600
    assert set(body["scope"].split()) == ALL


@pytest.mark.parametrize(
    ("data", "status", "error"),
    [
        ({"grant_type": "password"}, 400, "unsupported_grant_type"),
        ({"grant_type": "client_credentials", "client_id": "admin", "client_secret": "x"}, 401,
         "invalid_client"),
        ({"grant_type": "client_credentials", "client_id": "nope", "client_secret": "x"}, 401,
         "invalid_client"),
        ({"grant_type": "client_credentials", "client_id": "reader", "client_secret": "r3ad",
          "scope": "persons:delete"}, 400, "invalid_scope"),
    ],
)
def test_token_errors(data: dict[str, str], status: int, error: str) -> None:
    resp = make_app(make_tokens()).post("/oauth/token", data=data)
    assert resp.status_code == status
    assert resp.json()["error"] == error


def test_routes_require_valid_token() -> None:
    c = make_app(make_tokens())
    assert c.post("/identify", files=upload(b"A1")).status_code == 401
    bad = {"Authorization": "Bearer abc.def.ghi"}
    assert c.post("/identify", files=upload(b"A1"), headers=bad).status_code == 401

    forged = TokenService(AuthConfig(), "x" * 32, [OAuthClient("admin", "s3cret", ALL)])
    tok = forged.issue("admin", "s3cret", None)["access_token"]
    resp = c.post("/identify", files=upload(b"A1"), headers={"Authorization": f"Bearer {tok}"})
    assert resp.status_code == 401


def test_expired_token_rejected() -> None:
    import jwt

    cfg = AuthConfig()
    claims = {"iss": cfg.issuer, "aud": cfg.audience, "sub": "admin", "scope": "identify",
              "iat": 1_000, "exp": 2_000}
    token = jwt.encode(claims, SIGNING_KEY, algorithm="HS256")
    resp = make_app(make_tokens()).post(
        "/identify", files=upload(b"A1"), headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Token expirado"


def test_scope_enforced() -> None:
    c = make_app(make_tokens())
    headers = bearer(c, "reader", "r3ad")
    assert c.post("/identify", files=upload(b"A1"), headers=headers).status_code == 200
    assert c.post("/persons", files=upload(b"A1"), headers=headers).status_code == 403
    assert c.delete("/persons/123", headers=headers).status_code == 403


def test_auth_disabled() -> None:
    container = build_container(
        Settings(target="memory"), encoder=FakeEncoder(), repository=InMemoryFaceRepository()
    )
    c = TestClient(create_app(container, AuthConfig(enabled=False)))
    assert c.post("/identify", files=upload(b"A1")).status_code == 200
    assert c.post("/oauth/token", data={"grant_type": "client_credentials"}).status_code == 404
