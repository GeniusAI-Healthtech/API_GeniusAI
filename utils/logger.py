from loguru import logger
import sys


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")


setup_logger()


def get_logger():
    return logger
