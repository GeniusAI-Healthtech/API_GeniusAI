from PIL import Image
import io
from loguru import logger


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Converte a imagem de bytes para o formato PIL RGB

    Args:
        binary_image (bytes): A representação binária da imagem

    Returns:
        PIL.Image: A imagem no formato PIL RGB
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    logger.info("Imagem recebida e convertida em RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Converte a imagem PIL para bytes

    Args:
    image (Image): Uma instância de imagem PIL

    Returns:
    bytes : Objeto BytesIO que contém a imagem no formato JPEG com qualidade 85
    """
    return_image = io.BytesIO()
    image.save(
        return_image, format="JPEG", quality=85
    )  # salva a imagem em formato JPEG com qualidade 85
    return_image.seek(0)  # define o ponteiro para o início do arquivo
    return return_image
