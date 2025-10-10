import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from source.operator.insert_embeddings_to_vdb import run

from utils.helpers import create_logger
# Initialize FastAPI app
routes = APIRouter()
logger = create_logger()

@routes.get("/insert_to_vdb", response_class=JSONResponse)
async def insert_image_to_vdb(
  model_name: str = Query("resnet", description="Type of the embedder to use"),
  vdb_name: str = Query("milvus", description="Name of the vector database to use"),
  collection_name: str = Query(..., description="Name of the collection to insert embeddings into"),
):
  try: 
    run(model=model_name, vdb=vdb_name, collection_name=collection_name)
    return JSONResponse(content={"message": "Insertion proceses completed successfully"})
  except Exception as e:
    logger.error(f"Insertion process failed: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Insertion process failed: {str(e)}")

