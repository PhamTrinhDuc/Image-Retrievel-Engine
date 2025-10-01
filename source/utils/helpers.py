from loguru import logger
from datetime import datetime

def create_logger(job_name: str): 
  logger.add(
    f"logs/{job_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log",
    mode="a",
    format="{time} | {level} | {message}",
    rotation="5 MB",
    retention="7 days", 
  )

  return logger