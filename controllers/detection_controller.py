import json
from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import List
from starlette.responses import JSONResponse
from services.ai_services import detect_sample_model
from services.image_processing import add_bboxs_on_img
from utils.logger import get_logger
from utils.uses_for_images import get_image_from_bytes, encode_image_to_base64
from config import client

router = APIRouter()
logger = get_logger()

@router.post(
    "/{exam_type}/result_full",
    tags=["Analise"],
    summary="Retorna dados da análise, interpretação clínica e imagem com as detecções.",
)
async def complete_analysis(exam_type: str, files: List[UploadFile] = File(...)):
    """
    Object Detection, Clinical Interpretation and Annotated Image for multiple files.

    Args:
        exam_type (str): The type of exam being analyzed (ex: "ecg_signal").
        files (List[UploadFile]): List of image files in bytes format.

    Returns:
        list: List of JSON objects containing the Objects Detections, Clinical Interpretation, and Annotated Image for each file.
    """
    results = []
    try:
        for file in files:
            # Inicializa o dicionário de resultados
            result = {"data": None, "clinical_interpretation": None, "annotated_image": None}

            # Converte o arquivo de imagem para um objeto de imagem
            input_image = get_image_from_bytes(await file.read())

            # Predição do modelo
            predict = detect_sample_model(input_image, exam_type)

            # Seleciona as informações de detecção de objetos
            detect_res = predict[["name", "confidence"]]
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

            # Codifica a imagem em base64
            image_base64 = encode_image_to_base64(final_image)
            result["annotated_image"] = image_base64

            # Adiciona o resultado à lista de resultados
            results.append(result)

        # Log dos resultados e retorno
        logger.info("results: %s", results)

        return results

    except Exception as e:
        logger.error("Failed to process complete analysis: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing the complete analysis: {str(e)}")

@router.post(
    "/{exam_type}/result_object",
    tags=["Analise"],
    summary="Retorna os dados da análise",
)
async def img_object_detection_to_json(exam_type: str, files: List[UploadFile] = File(...)):
    """
    Object Detection from multiple images.

    Args:
        exam_type (str): The type of exam being analyzed (ex: "ecg_signal").
        files (List[UploadFile]): List of image files in bytes format.

    Returns:
        list: List of JSON objects containing the Objects Detections for each file.
    """
    results = []
    try:
        for file in files:
            # Inicializa o dicionário de resultados
            result = {"detect_objects": None}

            # Converte o arquivo de imagem para um objeto de imagem
            input_image = get_image_from_bytes(await file.read())

            # Predição do modelo
            predict = detect_sample_model(input_image, exam_type)

            # Seleciona as informações de detecção de objetos
            detect_res = predict[["name", "confidence"]]
            objects = detect_res["name"].values

            result["detect_objects_names"] = ", ".join(objects)
            result["detect_objects"] = json.loads(detect_res.to_json(orient="records"))

            # Adiciona o resultado à lista de resultados
            results.append(result)

        # Log dos resultados e retorno
        logger.info("results: %s", results)

        return results

    except Exception as e:
        logger.error("Failed to process image for object detection: %s", str(e))
        raise HTTPException(status_code=500, detail="Error processing the image")

@router.post(
    "/{exam_type}/result_img",
    tags=["Analise"],
    summary="Gera uma imagem com objetos detectados anotados.",
)
async def img_object_detection_to_img(exam_type: str, files: List[UploadFile] = File(...)):
    """
    Object Detection from multiple images and plot bbox on images.

    Args:
        exam_type (str): The type of exam being analyzed (ex: "ecg_signal").
        files (List[UploadFile]): List of image files in bytes format.

    Returns:
        list: List of images in bytes with bbox annotations.
    """
    try:
        result_images = []

        for file in files:
            try:
                # Converte o arquivo de imagem para um objeto de imagem
                logger.info(f"Processing file: {file.filename}")
                input_image = get_image_from_bytes(await file.read())

                # Predição do modelo
                predict = detect_sample_model(input_image, exam_type)
                logger.info(f"Prediction for {file.filename}: {predict}")

                # Adiciona as bounding boxes na imagem
                final_image = add_bboxs_on_img(image=input_image, predict=predict)

                # Codifica a imagem em base64
                image_base64 = encode_image_to_base64(final_image)
                result_images.append(image_base64)

            except Exception as file_error:
                logger.error(f"Failed to process file {file.filename}: {str(file_error)}")
                raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")

        # Log dos resultados e retorno
        logger.info("results: %s", result_images)
        return JSONResponse(content=result_images)

    except Exception as e:
        logger.error("Failed to annotate images with bounding boxes: %s", str(e))
        raise HTTPException(status_code=500, detail="Error annotating the images")

@router.post(
    "/{exam_type}/result_interpretation",
    tags=["Analise"],
    summary="Retorna apenas a interpretação clínica dos dados fornecidos.",
)
async def clinical_interpretation(exam_type: str, files: List[UploadFile] = File(...)):
    """
    Clinical Interpretation for multiple files.

    Args:
        exam_type (str): The type of exam being analyzed (ex: "ecg_signal").
        files (List[UploadFile]): List of image files in bytes format.

    Returns:
        list: List of JSON objects containing the Clinical Interpretation for each file.
    """
    results = []
    try:
        for file in files:
            # Converte o arquivo de imagem para um objeto de imagem
            input_image = get_image_from_bytes(await file.read())

            # Predição do modelo
            predict = detect_sample_model(input_image, exam_type)

            # Seleciona as informações de detecção de objetos
            detect_res = predict[["name", "confidence"]]
            detect_objects_json = json.loads(detect_res.to_json(orient="records"))

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
            result = {
                "exam_type": exam_type,
                "clinical_interpretation": message_content
            }

            # Adiciona o resultado à lista de resultados
            results.append(result)

        # Log dos resultados e retorno
        logger.info("results: %s", results)

        return results

    except Exception as e:
        logger.error("Failed to process clinical interpretation: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing the clinical interpretation: {str(e)}")
