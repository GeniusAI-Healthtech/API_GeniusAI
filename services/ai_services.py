from PIL import Image
import io
import pandas as pd


from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger()

ai_paths = {
    "ecg_signal": "./AI/cardiac/ecg-signal-detec.pt",  # Ajustado para claridade
    "ecg_ialjy": "./AI/cardiac/ecg-ialjy.pt",
    "best": "./AI/cardiac/best.pt",
    "stenosis": "./AI/cardiac/stenosis-uc0rf-yolo8.pt",
    "ecg_v3": "./AI/cardiac/ecg-v3.pt",
    "ecg_lead_classification": "./AI/cardiac/ecg-lead-classification.pt",
    "ecg_labeled_marzo": "./AI/cardiac/ecg_labeled_marzo.pt",
}


# Função para carregar modelos dinamicamente
def load_model(exam_type: str):
    if exam_type in ai_paths:
        model_path = ai_paths[exam_type]
        model = YOLO(model_path)
        logger.info("Modelo carregado: {}", model_path)
        return model
    else:
        logger.error("Tipo de exame/modelo não suportado: {}", exam_type)
        raise ValueError("Modelo não suportado")


def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transforma a previsão do YOLOv8 (torch.Tensor) em um DataFrame do pandas.

    Args:
        results (list): Uma lista contendo a saída da previsão do YOLOv8 na forma de um torch.Tensor.
        labeles_dict (dict): Um dicionário contendo os nomes dos rótulos, onde as chaves são os IDs das classes e os valores são os nomes dos rótulos.

    Returns:
        predict_bbox (pd.DataFrame): Um DataFrame contendo as coordenadas da bounding box, os escores de confiança e os rótulos das classes.
    """
    # Transforma o Tensor para um array numpy
    predict_bbox = pd.DataFrame(
        results[0].to("cpu").numpy().boxes.xyxy,
        columns=["xmin", "ymin", "xmax", "ymax"],
    )
    # Adiciona a confiança da previsão ao DataFrame
    predict_bbox["confidence"] = results[0].to("cpu").numpy().boxes.conf
    # Adiciona a classe da previsão ao DataFrame
    predict_bbox["class"] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Substitui o número da classe pelo nome da classe a partir do labeles_dict
    predict_bbox["name"] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox


def get_model_predict(
    model: YOLO,
    input_image: Image,
    save: bool = False,
    image_size: int = 1248,
    conf: float = 0.5,
    augment: bool = False,
) -> pd.DataFrame:
    """
    Obtém as previsões de um modelo em uma imagem de entrada.

    Args:
        model (YOLO): O modelo YOLO treinado.
        input_image (Image): A imagem na qual o modelo fará previsões.
        save (bool, opcional): Se deve salvar a imagem com as previsões. Padrão é False.
        image_size (int, opcional): O tamanho da imagem que o modelo receberá. Padrão é 1248.
        conf (float, opcional): O limiar de confiança para as previsões. Padrão é 0.5.
        augment (bool, opcional): Se deve aplicar aumento de dados na imagem de entrada. Padrão é False.

    Returns:
        pd.DataFrame: Um DataFrame contendo as previsões.
    """
    # Faz as previsões
    logger.info(
        "Parâmetros de entrada para predict: image_size={}, conf={}, save={}, augment={}",
        image_size,
        conf,
        save,
        augment,
    )
    predictions = model.predict(
        imgsz=image_size,
        source=input_image,
        conf=conf,
        save=save,
        augment=augment,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
    )

    # Transforma as previsões em um dataframe do pandas
    logger.info("Previsões brutas retornadas do modelo: {}", predictions)
    return transform_predict_to_df(predictions, model.model.names)


def detect_sample_model(input_image: Image, exam_type: str) -> pd.DataFrame:
    model = load_model(exam_type)  # Carrega o modelo com base no tipo de exame
    # Chama get_model_predict com parâmetros específicos
    return get_model_predict(
        model, input_image, save=False, image_size=640, conf=0.5, augment=False
    )
