"""Adaptador FaceEncoder baseado em InsightFace (ArcFace, 512 dimensões)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from booprecognition.domain.errors import InvalidImageError, NoFaceDetectedError
from booprecognition.domain.value_objects import Embedding


class InsightFaceEncoder:
    """Detecta o maior rosto da imagem e devolve seu embedding ArcFace."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)

    def encode(self, image: bytes) -> Embedding:
        frame = self._decode(image)
        faces: list[Any] = self._app.get(frame)
        if not faces:
            raise NoFaceDetectedError("Nenhum rosto detectado na imagem")
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return Embedding.from_iterable(largest.normed_embedding.tolist())

    @staticmethod
    def _decode(image: bytes) -> np.ndarray:
        frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise InvalidImageError("Imagem inválida ou formato não suportado")
        return frame
