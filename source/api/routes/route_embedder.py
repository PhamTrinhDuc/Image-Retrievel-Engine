import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from models.base import SearchResponse, HealthResponse
from embedder.extractor_factory import EmbedderFactory
from configs.helper import DEFAULT_EXTRACTOR_CONFIGS

from utils.helpers import create_logger

# Initialize FastAPI app
routes = APIRouter()
logger = create_logger()


@routes.get("/list_embedders", response_model=None)
async def list_embedders():
   logger.info("Listing supported embedders")
   return JSONResponse(content={"supported_embedders": list(DEFAULT_EXTRACTOR_CONFIGS.keys())})

@routes.get("/embedder_info", response_model=None)
async def get_embedder_info():
  """
  Get information about the embedder.
  """ 
  return JSONResponse(content={"embedder_info": DEFAULT_EXTRACTOR_CONFIGS})

@routes.get("/health", response_model=HealthResponse)
async def health_check(extractor_type: str = Query("resnet", description="Type of the embedder to check")):
  """
  Health check endpoint to verify if the service is running.
  """
  if extractor_type not in DEFAULT_EXTRACTOR_CONFIGS:
      return HealthResponse(status="error", code=404, message=f"Unsupported embedder type: {extractor_type}. Supported embedders are: {list(DEFAULT_EXTRACTOR_CONFIGS.keys())}")
  try:
      embedder = EmbedderFactory.create_extractor(extractor_type=extractor_type)
      logger.info(f"Health check successful for embedder: {extractor_type}")
      return HealthResponse(status="ok", code=200, message=f"Service is running with embedder: {extractor_type}")
  except Exception as e:
      logger.error(f"Health check failed: {str(e)}")
      return HealthResponse(status="error", code=404, message=f"Health check failed: {str(e)}")

