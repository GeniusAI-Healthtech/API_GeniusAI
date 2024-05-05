####################################### logger #################################

from loguru import logger
import sys
from functools import wraps

def setup_logger():
    logger.remove()  # Remove all handlers associated with the logger
    logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

setup_logger()

def get_logger():
    return logger

logger = get_logger()

def log_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"🚀 Iniciando {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ {func.__name__} concluído com sucesso!")
            return result
        except Exception as e:
            logger.error(f"Erro em {func.__name__}: {str(e)}")
            raise e
    return wrapper
