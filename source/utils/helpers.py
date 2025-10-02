from loguru import logger
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def create_logger(job_name: str):
    """
    Auto-detect mode via file .env:
      - If ENV=dev  -> log to console
      - If ENV=prod -> log to file
    """
    env = os.getenv("ENVIROMENT_LOG", "dev")  # mặc định dev
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{extra[job]} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    if env == "dev":
        logger.add(sys.stdout, format=log_format, level="DEBUG")
    else:
        # log ra file
        os.makedirs("logs", exist_ok=True)
        logger.add(
            f"logs/{job_name}.log",
            format=log_format,
            rotation="1 day",     # create new file per day
            retention="7 days",   # keep file log in 7 days
            level="INFO",
            encoding="utf-8"
        )

    return logger.bind(job=job_name)
