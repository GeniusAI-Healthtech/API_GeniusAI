import json
import base64
from fastapi import APIRouter, File, HTTPException
from fastapi.responses import StreamingResponse
from services.ai_services import detect_sample_model
from services.image_processing import add_bboxs_on_img
from utils.logger import get_logger
from utils.image_utils import get_image_from_bytes, get_bytes_from_image
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

# Recupera a chave da API de um arquivo .env
api_key = os.getenv("OPENAI_API_KEY")

# Inicializa o cliente com a chave da API
client = OpenAI(api_key=api_key)

router = APIRouter()
logger = get_logger()

@router.post(
    "/{exam_type}/result_full",
    tags=["Analise"],
    summary="Retorna dados da análise, interpretação clínica e imagem com as detecções.",
)
async def complete_analysis(exam_type: str, file: bytes = File(...)):
    """
    Object Detection, Clinical Interpretation and Annotated Image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections, Clinical Interpretation and Annotated Image.
    """
    try:
        # Inicializa o dicionário de resultados
        result = {"data": None, "clinical_interpretation": None, "annotated_image": None}

        # Converte o arquivo de imagem para um objeto de imagem
        input_image = get_image_from_bytes(file)

        # Predição do modelo
        predict = detect_sample_model(input_image, exam_type)

        # Seleciona as informações de detecção de objetos
        detect_res = predict[["name", "confidence"]]
        objects = detect_res["name"].values

        detect_objects_json = json.loads(detect_res.to_json(orient="records"))
        result["data"] = {
            "exam_type": exam_type,
            "analysis_results": detect_objects_json
        }

        # Obtem a interpretação clínica do GPT
        prompt = (
            "Descreva em um único parágrafo simples a interpretação clínica dos dados fornecidos. "
            "Faça isso em português. "
            "O retorno deve ser um único parágrafo. "
            "Dados: " + str(detect_objects_json)
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Output the response as a single, simple paragraph in Portuguese."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )

        message_content = response.choices[0].message.content
        result["clinical_interpretation"] = message_content

        # Adiciona as bounding boxes na imagem
        final_image = add_bboxs_on_img(image=input_image, predict=predict)

        # Converte a imagem final para bytes
        image_bytes = get_bytes_from_image(final_image)

        # Codifica a imagem em base64
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        result["annotated_image"] = image_base64

        # Log dos resultados e retorno
        logger.info("results: %s", result)

        return result

    except Exception as e:
        logger.error("Failed to process complete analysis: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing the complete analysis: {str(e)}")

@router.post(
    "/{exam_type}/result_object",
    tags=["Analise"],
    summary="Retorna os dados da análise",
)
def img_object_detection_to_json(exam_type: str, file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    try:
        # Inicializa o dicionário de resultados
        result = {"detect_objects": None}

        # Converte o arquivo de imagem para um objeto de imagem
        input_image = get_image_from_bytes(file)

        # Predição do modelo
        predict = detect_sample_model(input_image, exam_type)

        # Seleciona as informações de detecção de objetos
        detect_res = predict[["name", "confidence"]]
        objects = detect_res["name"].values

        result["detect_objects_names"] = ", ".join(objects)
        result["detect_objects"] = json.loads(detect_res.to_json(orient="records"))

        # Log dos resultados e retorno
        logger.info("results: %s", result)

        return result

    except Exception as e:
        logger.error("Failed to process image for object detection: %s", str(e))
        raise HTTPException(status_code=500, detail="Error processing the image")

@router.post(
    "/{exam_type}/result_img",
    tags=["Analise"],
    summary="Gera uma imagem com objetos detectados anotados.",
)
def img_object_detection_to_img(exam_type: str, file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    try:
        # Converte o arquivo de imagem para um objeto de imagem
        input_image = get_image_from_bytes(file)

        # Predição do modelo
        predict = detect_sample_model(input_image, exam_type)

        # Adiciona as bounding boxes na imagem
        final_image = add_bboxs_on_img(image=input_image, predict=predict)

        # Retorna a imagem em formato de bytes
        logger.info("results: %s", final_image)
        return StreamingResponse(
            content=get_bytes_from_image(final_image), media_type="image/jpeg"
        )
    except Exception as e:
        logger.error("Failed to annotate image with bounding boxes: %s", str(e))
        raise HTTPException(status_code=500, detail="Error annotating the image")
