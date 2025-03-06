from PIL import Image
import pandas as pd
import numpy as np
from ultralytics.utils.plotting import colors, Annotator
from utils.uses_for_images import get_image_from_bytes, get_bytes_from_image
from utils.logger import get_logger
from fastapi.exceptions import HTTPException

logger = get_logger()


def crop_image_by_predict(
    image: Image,
    predict: pd.DataFrame,
    crop_class_name: str,
) -> Image:
    """Crop an image based on the detection of a certain object in the image.

    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.

    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict["name"] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(
            status_code=400, detail=f"{crop_class_name} not found in photo"
        )

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=["confidence"], ascending=False)

    crop_bbox = crop_predicts[["xmin", "ymin", "xmax", "ymax"]].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return img_crop


################################# Função de Bounding Box #####################################
def add_bboxs_on_img(image: Image, predict: pd.DataFrame) -> Image:
    """
    Adiciona uma bounding box na imagem

    Args:
    image (Image): Imagem de entrada
    predict (pd.DataFrame): Previsão do modelo

    Returns:
    Image: Imagem com as bounding boxes
    """
    # Cria um objeto annotator
    annotator = Annotator(np.array(image))

    # Ordena a previsão pelo valor xmin
    predict = predict.sort_values(by=["xmin"], ascending=True)

    # Itera sobre as linhas do dataframe de previsão
    for i, row in predict.iterrows():
        # Cria o texto a ser exibido na imagem
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # Obtém as coordenadas da bounding box
        bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
        # Adiciona a bounding box e o texto na imagem
        annotator.box_label(bbox, text, color=colors(row["class"], True))
    # Converte a imagem anotada para imagem PIL
    logger.info("Bounding boxes adicionadas à imagem")
    return Image.fromarray(annotator.result())
