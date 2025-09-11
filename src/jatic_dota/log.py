import logging

logger = logging.getLogger("jatic_dota")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger('fvcore').setLevel(logging.ERROR)

if not logger.hasHandlers():
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
