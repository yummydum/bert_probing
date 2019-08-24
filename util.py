from pathlib import Path
import logging
from logging import getLogger, StreamHandler, FileHandler
import requests


def set_logger(name: str):
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(process)d-%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%d-%b-%y %H:%M:%S')

    # Logging to stdout
    s_handler = StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)

    # Logging to file
    log_file_path = Path(f"log/{name}.log")
    f_handler = FileHandler(log_file_path)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)

    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)
    return logger


def notify_slack(message):
    webhook = f"https://hooks.slack.com/services/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"
    message = json.dumps({"text": f"{message}"})
    requests.post(webhook, message)
    return
