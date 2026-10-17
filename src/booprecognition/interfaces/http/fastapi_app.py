"""Entrega HTTP (FastAPI). Só traduz HTTP <-> casos de uso."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.responses import JSONResponse

from booprecognition.config.container import Container, build_container
from booprecognition.config.settings import AuthConfig, load_settings
from booprecognition.domain.errors import (
    InvalidImageError,
    NoFaceDetectedError,
    PersonNotFoundError,
)
from booprecognition.interfaces.http.auth import (
    OAuthError,
    TokenService,
    scope_guard,
    token_endpoint,
)


def create_app(
    container: Container | None = None,
    auth: AuthConfig | TokenService | None = None,
) -> FastAPI:
    """`auth`: config ou TokenService pronto; se omitido, vem de `auth` no YAML."""
    app = FastAPI(title="booprecognition", version="0.1.0")
    if container is None or auth is None:
        settings = load_settings()
        container = container or build_container(settings)
        auth = auth or settings.auth
    services = container
    tokens: TokenService | None
    if isinstance(auth, AuthConfig):
        tokens = TokenService.from_config(auth) if auth.enabled else None
    else:
        tokens = auth
    guard = scope_guard(tokens)

    @app.exception_handler(OAuthError)
    async def _oauth(_: Request, exc: OAuthError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Basic"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error, "error_description": exc.description},
            headers={**(headers or {}), "Cache-Control": "no-store"},
        )

    if tokens is not None:
        app.post("/oauth/token", tags=["auth"])(token_endpoint(tokens))

    @app.exception_handler(NoFaceDetectedError)
    @app.exception_handler(InvalidImageError)
    async def _unprocessable(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PersonNotFoundError)
    async def _not_found(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.post("/identify")
    async def identify(
        image: UploadFile = File(...), _: Any = Security(guard, scopes=["identify"])
    ) -> dict[str, Any] | None:
        result = services.identify_person.execute(await _read(image))
        return result.to_dict() if result else None

    @app.post("/persons")
    async def register(
        image: UploadFile = File(...), _: Any = Security(guard, scopes=["persons:write"])
    ) -> JSONResponse:
        result = services.register_person.execute(await _read(image))
        body = result.to_dict()
        headers = {"Location": f"/persons/{body['id']}"} if result.created else None
        status = 201 if result.created else 200
        return JSONResponse(status_code=status, content=body, headers=headers)

    @app.delete("/persons/{person_id}")
    async def remove(
        person_id: int, _: Any = Security(guard, scopes=["persons:delete"])
    ) -> dict[str, Any]:
        return services.remove_person.execute(person_id).to_dict()

    return app


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Imagem vazia")
    return data
