"""Adaptador FaceEncoder baseado em InsightFace (ArcFace).

Com um modelo o vetor tem 512 dimensões; com vários (ensemble) os vetores de
cada modelo são normalizados e concatenados (512 × N dimensões).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from booprecognition.domain.errors import (
    InvalidImageError,
    LowQualityFaceError,
    NoFaceDetectedError,
)
from booprecognition.domain.value_objects import Embedding

# Lado maior máximo antes da detecção: fotos de celular (4000px+) deixam o
# rosto desproporcional ao det_size e só aumentam o custo.
_MAX_SIDE = 1600
# Imagens muito pequenas são ampliadas para o detector enxergar o rosto.
_MIN_SIDE = 320

_Variant = Callable[[np.ndarray], np.ndarray]
_Face = tuple[np.ndarray, np.ndarray]  # (bbox + score, 5 landmarks)


class InsightFaceEncoder:
    """Detecta o maior rosto da imagem e devolve seu embedding ArcFace.

    A detecção é feita em várias tentativas, da mais barata para a mais cara,
    parando na primeira que encontrar um rosto: imagem original, imagem com
    borda (rostos em close que encostam nas bordas), contraste equalizado
    (fotos escuras/contraluz), ampliação 2x (rostos pequenos) e rotações de
    90°/180°/270° (fotos sem orientação EXIF correta). As tentativas extras
    usam um limiar de confiança menor que o da primeira.

    Para usar mais características do rosto e reduzir falsos positivos:
    - ensemble: cada modelo em `model_names` (ex.: buffalo_l = ResNet50/WebFace600K,
      antelopev2 = ResNet100/Glint360K) gera o próprio vetor e eles são
      concatenados com peso igual, então o cosseno final é a média dos cossenos;
    - o rosto alinhado e sua versão espelhada são somados (test-time augmentation);
    - rostos com baixa confiança de detecção ou muito de perfil são recusados,
      porque geram vetores pouco discriminativos que "parecem" com todo mundo.
    """

    def __init__(
        self,
        model_names: Sequence[str] = ("buffalo_l",),
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        fallback_det_thresh: float = 0.3,
        min_face_size: int = 40,
        min_det_score: float = 0.6,
        max_yaw_ratio: float = 0.45,
    ) -> None:
        if not model_names:
            raise ValueError("Informe ao menos um modelo de reconhecimento")
        providers = providers or ["CPUExecutionProvider"]
        self._app = FaceAnalysis(
            name=model_names[0],
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        self._recognizers = [self._app.models["recognition"]]
        for name in model_names[1:]:
            extra = FaceAnalysis(
                name=name, allowed_modules=["detection", "recognition"], providers=providers
            )
            extra.prepare(ctx_id=0, det_size=det_size)
            self._recognizers.append(extra.models["recognition"])
        self._det_thresh = det_thresh
        self._fallback_det_thresh = fallback_det_thresh
        self._min_face_size = min_face_size
        self._min_det_score = min_det_score
        self._max_yaw_ratio = max_yaw_ratio

    def encode(self, image: bytes) -> Embedding:
        frame = self._normalize_size(self._decode(image))
        try:
            for index, variant in enumerate(self._variants()):
                self._app.det_model.det_thresh = (
                    self._det_thresh if index == 0 else self._fallback_det_thresh
                )
                candidate = variant(frame)
                face = self._largest_face(*self._app.det_model.detect(candidate, max_num=0))
                if face is not None:
                    self._check_quality(face)
                    return self._embed(candidate, face[1])
        finally:
            self._app.det_model.det_thresh = self._det_thresh
        raise NoFaceDetectedError(
            "Nenhum rosto detectado na imagem. Use uma foto frontal, bem iluminada "
            "e com o rosto inteiro visível."
        )

    def _embed(self, frame: np.ndarray, kps: np.ndarray) -> Embedding:
        weight = 1.0 / math.sqrt(len(self._recognizers))
        parts: list[np.ndarray] = []
        for rec in self._recognizers:
            crop = face_align.norm_crop(frame, landmark=kps, image_size=rec.input_size[0])
            vector = rec.get_feat([crop, cv2.flip(crop, 1)]).sum(axis=0)
            parts.append(vector / np.linalg.norm(vector) * weight)
        return Embedding.from_iterable(np.concatenate(parts).tolist())

    def _check_quality(self, face: _Face) -> None:
        bbox, kps = face
        if float(bbox[4]) < self._min_det_score:
            raise LowQualityFaceError(
                "Rosto pouco nítido para reconhecimento. Use uma foto frontal e bem iluminada."
            )
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        if eye_distance == 0.0:
            raise LowQualityFaceError("Rosto inválido para reconhecimento")
        # Deslocamento horizontal do nariz em relação ao centro dos olhos: ~0 de frente.
        yaw = abs(float(nose[0] - (left_eye[0] + right_eye[0]) / 2)) / eye_distance
        if yaw > self._max_yaw_ratio:
            raise LowQualityFaceError(
                "Rosto muito de perfil. Olhe para a câmera e tente novamente."
            )

    def _largest_face(self, bboxes: np.ndarray, kpss: np.ndarray | None) -> _Face | None:
        if kpss is None:
            return None
        best: _Face | None = None
        best_area = 0.0
        for bbox, kps in zip(bboxes, kpss, strict=True):
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if min(width, height) >= self._min_face_size and width * height > best_area:
                best, best_area = (bbox, kps), float(width * height)
        return best

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
