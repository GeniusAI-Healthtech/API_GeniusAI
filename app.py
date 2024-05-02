from PIL import Image
import io
import pandas as pd
import numpy as np
from loguru import logger

from typing import Optional

from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator


# Inicializa os modelos
# ecg-signal-detec
# ecg-ialjy.pt 
# best.pt
# stenosis-uc0rf-yolo8.pt
# ecg-v3.pt
# ecg-lead-classification.pt
# ecg_labeled_marzo.pt

model_sample_model = YOLO("./AI/cardiac/ecg-signal-detec.pt")
logger.info("Modelo carregado: {}", model_sample_model.model_name)


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
    image.save(return_image, format='JPEG', quality=85)  # salva a imagem em formato JPEG com qualidade 85
    return_image.seek(0)  # define o ponteiro para o início do arquivo
    return return_image

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
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Adiciona a confiança da previsão ao DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Adiciona a classe da previsão ao DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Substitui o número da classe pelo nome da classe a partir do labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
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
    logger.info("Parâmetros de entrada para predict: image_size={}, conf={}, save={}, augment={}", image_size, conf, save, augment)

    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    logger.info("Previsões brutas retornadas do modelo: {}", predictions)
    
    # Transforma as previsões em um dataframe do pandas
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


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
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # Itera sobre as linhas do dataframe de previsão
    for i, row in predict.iterrows():
        # Cria o texto a ser exibido na imagem
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # Obtém as coordenadas da bounding box
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # Adiciona a bounding box e o texto na imagem
        annotator.box_label(bbox, text, color=colors(row['class'], True))
    # Converte a imagem anotada para imagem PIL
    logger.info("Bounding boxes adicionadas à imagem")
    return Image.fromarray(annotator.result())


################################# Modelos #####################################


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Faz a previsão usando o modelo de amostra.
    Baseado no YOLOv8

    Args:
        input_image (Image): A imagem de entrada.

    Returns:
        pd.DataFrame: DataFrame contendo a localização do objeto.
    """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    logger.info("Previsões do modelo: {}", predict)
    return predict