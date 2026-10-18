"""Erros de domínio."""


class DomainError(Exception):
    """Base para todos os erros de domínio."""


class NoFaceDetectedError(DomainError):
    """Nenhum rosto foi encontrado na imagem."""


class LowQualityFaceError(NoFaceDetectedError):
    """Rosto encontrado, mas sem qualidade para reconhecimento (borrado, de perfil)."""


class InvalidImageError(DomainError):
    """Os bytes recebidos não representam uma imagem válida."""


class PersonNotFoundError(DomainError):
    def __init__(self, person_id: int) -> None:
        super().__init__(f"Pessoa {person_id} não encontrada")
        self.person_id = person_id


class InvalidEmbeddingError(DomainError):
    """Vetor de embedding inválido (vazio ou nulo)."""
