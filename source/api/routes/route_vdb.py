import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from models.base import SearchResponse, HealthResponse
from vector_db.vdb_factory import VectorDBFactory
from configs.helper import DEFAULT_SEARCH_CONFIGS

from utils.helpers import create_logger

# Initialize logger
logger = create_logger("route_vdb")

# Initialize FastAPI app
routes = APIRouter()

@routes.get("/list_vdb", response_class=JSONResponse)
async def list_vector_db():
   return JSONResponse(content={"supported_vector_dbs": list(DEFAULT_SEARCH_CONFIGS.keys())})
      

@routes.get("/health", response_model=HealthResponse)
async def health_check(vdb_name: str = Query("milvus", description="Name of the vector database to check")):
  """
  Health check endpoint to verify if the service is running.
  """
  if vdb_name not in DEFAULT_SEARCH_CONFIGS:
        return HealthResponse(status="error", message=f"Unsupported vector database: {vdb_name}. Supported databases are: {list(DEFAULT_SEARCH_CONFIGS.keys())}")
  try:
        vdb = VectorDBFactory.create_client(db_type=vdb_name)
        vdb.connect()
        return HealthResponse(status="ok", code=200, message="Service is running")
  except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(status="error", code=500, message=f"Health check failed: {str(e)}")


@routes.get("/get_vdb_info", response_class=JSONResponse)
async def get_vector_database_info(vdb_name: str = Query("milvus", description="Name of the vector database to get info for")):
  """
  Get information about the vector database.
  """
  if vdb_name not in DEFAULT_SEARCH_CONFIGS:
      return JSONResponse(status_code=400, content={"error": f"Unsupported vector database: {vdb_name}. Supported databases are: {list(DEFAULT_SEARCH_CONFIGS.keys())}"})
  
  return JSONResponse(content={"vector_db_info": DEFAULT_SEARCH_CONFIGS[vdb_name]})


@routes.get("/list_collections", response_class=JSONResponse)
async def list_vector_collections():
  """
  List all available vector collections.
  """
  try:
      vdb = VectorDBFactory.create_client(db_type="milvus")
      vdb.connect()
      db_list = vdb.list_collections()
      return JSONResponse(content={"collections": db_list})
  except Exception as e:
      logger.error(f"Failed to list vector databases: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Failed to list vector databases: {str(e)}")


@routes.get("/collection_stats", response_class=JSONResponse)
async def get_collection_stats(vdb_name: str = Query("milvus", description="Name of the vector database"), 
                               collection_name: str = Query(..., description="Name of the collection to get stats for")):
  """
  Get statistics of a specific collection.
  """
  try:
      vdb = VectorDBFactory.create_client(db_type=vdb_name)
      stats = vdb.get_collection_stats(collection_name=collection_name)
      return JSONResponse(content={"collection_stats": stats})
  except Exception as e:
      logger.error(f"Failed to get collection stats: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {str(e)}")

