"""Entrega HTTP (FastAPI). Só traduz HTTP <-> casos de uso."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from booprecognition.config.container import Container, build_container
from booprecognition.domain.errors import (
    InvalidImageError,
    NoFaceDetectedError,
    PersonNotFoundError,
)


def create_app(container: Container | None = None) -> FastAPI:
    app = FastAPI(title="booprecognition", version="0.1.0")
    services = container or build_container()

    @app.exception_handler(NoFaceDetectedError)
    @app.exception_handler(InvalidImageError)
    async def _unprocessable(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PersonNotFoundError)
    async def _not_found(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.post("/identify")
    async def identify(image: UploadFile = File(...)) -> dict[str, Any] | None:
        result = services.identify_person.execute(await _read(image))
        return result.to_dict() if result else None

    @app.post("/persons")
    async def register(image: UploadFile = File(...)) -> JSONResponse:
        result = services.register_person.execute(await _read(image))
        return JSONResponse(status_code=201 if result.created else 200, content=result.to_dict())

    @app.delete("/persons/{person_id}")
    async def remove(person_id: int) -> dict[str, Any]:
        return services.remove_person.execute(person_id).to_dict()

    return app


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Imagem vazia")
    return data
