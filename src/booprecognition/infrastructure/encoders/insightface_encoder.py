"""Adaptador FaceEncoder baseado em InsightFace (ArcFace, 512 dimensões)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from booprecognition.domain.errors import InvalidImageError, NoFaceDetectedError
from booprecognition.domain.value_objects import Embedding

# Lado maior máximo antes da detecção: fotos de celular (4000px+) deixam o
# rosto desproporcional ao det_size e só aumentam o custo.
_MAX_SIDE = 1600
# Imagens muito pequenas são ampliadas para o detector enxergar o rosto.
_MIN_SIDE = 320

_Variant = Callable[[np.ndarray], np.ndarray]


class InsightFaceEncoder:
    """Detecta o maior rosto da imagem e devolve seu embedding ArcFace.

    A detecção é feita em várias tentativas, da mais barata para a mais cara,
    parando na primeira que encontrar um rosto: imagem original, imagem com
    borda (rostos em close que encostam nas bordas), contraste equalizado
    (fotos escuras/contraluz), ampliação 2x (rostos pequenos) e rotações de
    90°/180°/270° (fotos sem orientação EXIF correta). As tentativas extras
    usam um limiar de confiança menor que o da primeira.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        fallback_det_thresh: float = 0.3,
        min_face_size: int = 40,
    ) -> None:
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        self._det_thresh = det_thresh
        self._fallback_det_thresh = fallback_det_thresh
        self._min_face_size = min_face_size

    def encode(self, image: bytes) -> Embedding:
        frame = self._normalize_size(self._decode(image))
        for index, variant in enumerate(self._variants()):
            self._app.det_model.det_thresh = (
                self._det_thresh if index == 0 else self._fallback_det_thresh
            )
            face = self._largest_face(self._app.get(variant(frame)))
            if face is not None:
                self._app.det_model.det_thresh = self._det_thresh
                return Embedding.from_iterable(face.normed_embedding.tolist())
        self._app.det_model.det_thresh = self._det_thresh
        raise NoFaceDetectedError(
            "Nenhum rosto detectado na imagem. Use uma foto frontal, bem iluminada "
            "e com o rosto inteiro visível."
        )

    def _largest_face(self, faces: list[Any]) -> Any | None:
        valid = [
            f for f in faces
            if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= self._min_face_size
        ]
        if not valid:
            return None
        return max(valid, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def _variants() -> Iterator[_Variant]:
        yield lambda f: f
        yield _pad
        yield _equalize
        yield lambda f: _pad(_equalize(f))
        yield lambda f: cv2.resize(f, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            yield lambda f, r=rotation: _pad(cv2.rotate(f, r))

    @staticmethod
    def _normalize_size(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        longest, shortest = max(height, width), min(height, width)
        if longest > _MAX_SIDE:
            scale = _MAX_SIDE / longest
            return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if shortest < _MIN_SIDE:
            scale = _MIN_SIDE / shortest
            return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return frame

    @staticmethod
    def _decode(image: bytes) -> np.ndarray:
        # IMREAD_COLOR já aplica a orientação EXIF da foto.
        frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise InvalidImageError("Imagem inválida ou formato não suportado")
        return frame


def _pad(frame: np.ndarray) -> np.ndarray:
    """Centraliza a imagem num quadrado com borda de 25%, afastando o rosto das bordas."""
    height, width = frame.shape[:2]
    side = int(max(height, width) * 1.25)
    top, left = (side - height) // 2, (side - width) // 2
    return cv2.copyMakeBorder(
        frame, top, side - height - top, left, side - width - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def _equalize(frame: np.ndarray) -> np.ndarray:
    """Equaliza o contraste (CLAHE) no canal de luminância."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
