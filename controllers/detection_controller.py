from fastapi import APIRouter, File, HTTPException
from services.ai_services import detect_sample_model
from utils.logger import get_logger
import json

from services.ai_services import detect_sample_model
from services.image_processing import add_bboxs_on_img
from utils.image_utils import get_image_from_bytes
from utils.image_utils import get_bytes_from_image
from fastapi.responses import StreamingResponse

router = APIRouter()
logger = get_logger()


@router.post(
    "/{exam_type}/resultado_json",
    tags=["Analise"],
    summary="Retorna detecção de objetos em uma imagem por json.",
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
        # Step 1: Initialize the result dictionary with None values
        result = {"detect_objects": None}

        # Step 2: Convert the image file to an image object
        input_image = get_image_from_bytes(file)

        # Step 3: Predict from model
        predict = detect_sample_model(input_image, exam_type)

        # Step 4: Select detect obj return info
        # here you can choose what data to send to the result
        detect_res = predict[["name", "confidence"]]
        objects = detect_res["name"].values

        result["detect_objects_names"] = ", ".join(objects)
        result["detect_objects"] = json.loads(detect_res.to_json(orient="records"))

        # Step 5: Logs and return
        logger.info("results: {}", result)

        return result

    except Exception as e:
        logger.error("Failed to process image for object detection: {}", str(e))
        raise HTTPException(status_code=500, detail="Error processing the image")


@router.post(
    "/{exam_type}/resultado_img",
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
        # get image from bytes
        input_image = get_image_from_bytes(file)

        # model predict
        predict = detect_sample_model(input_image, exam_type)

        # add bbox on image
        final_image = add_bboxs_on_img(image=input_image, predict=predict)

        # return image in bytes format
        logger.info("results: {}", final_image)
        return StreamingResponse(
            content=get_bytes_from_image(final_image), media_type="image/jpeg"
        )
    except Exception as e:
        logger.error("Failed to annotate image with bounding boxes: {}", str(e))
        raise HTTPException(status_code=500, detail="Error annotating the image")
