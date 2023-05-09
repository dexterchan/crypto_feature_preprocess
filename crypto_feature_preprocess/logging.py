import logging
import os
from typing import Any, Union
from pythonjsonlogger import jsonlogger
from cachetools import cached, LRUCache

LOG_FILE: str = os.environ.get("LOG_FILE", "info.log")
json_format_str: str = (
    "%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)s"
)
format_str: str = "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)"

# pythonjsonlogger reference: https://www.datadoghq.com/blog/python-logging-best-practices/#unify-all-your-python-logs
json_formatter = jsonlogger.JsonFormatter(
    json_format_str, datefmt="%Y-%m-%d %H:%M:%S %z"
)

formatter = logging.Formatter(format_str)

logging.basicConfig(format=format_str, level=logging.getLevelName("INFO"))


# stream_handler = logging.StreamHandler()
# json_formatter = jsonlogger.JsonFormatter()

# # stream_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)
# stream_handler.setLevel(logging.INFO)

# file_handler = logging.FileHandler("info.log")
# # file_handler.setFormatter(formatter)
# file_handler.setFormatter(json_formatter)
# file_handler.setLevel(logging.DEBUG)

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(stream_handler)
# logger.addHandler(file_handler)


@cached(cache=LRUCache(maxsize=1024 * 1024))
def get_logger(name: str, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """Produces a new logger with the application logging parameter preset.

    Args:
        name (str): [description]
        level (Union[str, int], optional): [description]. Defaults to LOG_LEVEL.

    Returns:
        logging.Logger: [description]
    """
    logger = logging.getLogger(name)
    _level = logging.getLevelName(level)
    logger.setLevel(_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(json_formatter)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(_level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(json_formatter)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(_level)
    logger.addHandler(file_handler)
    return logger


# logger = get_logger("crypto_trading")


@cached(cache=LRUCache(maxsize=1024 * 1024))
def get_test_logger(
    name: str, level: Union[str, int] = logging.DEBUG
) -> logging.Logger:
    """Produces a new logger with the application logging parameter preset.

    Args:
        name (str): [description]
        level (Union[str, int], optional): [description]. Defaults to LOG_LEVEL.

    Returns:
        logging.Logger: [description]
    """
    logger = logging.getLogger(name)
    _level = logging.getLevelName(level)
    logger.setLevel(_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(_level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(_level)
    logger.addHandler(file_handler)
    return logger
