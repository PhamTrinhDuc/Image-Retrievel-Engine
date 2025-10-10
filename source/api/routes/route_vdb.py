import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any

from models.base import HealthResponse
from vector_db.vdb_factory import VectorDBFactory
from configs.helper import DEFAULT_SEARCH_CONFIGS
from utils.helpers import create_logger

# Initialize router
routes = APIRouter()
logger = create_logger()

async def get_vdb_client(
    vdb_name: str = Query("milvus", description="Name of the vector database"),
    collection_name: str = Query("resnet_embedding", description="Collection name")
):
    """Create and return connected VDB client."""
    try:
        vdb = VectorDBFactory.create_client(vdb_name=vdb_name, **{"collection_name": collection_name})
        vdb.connect()
        vdb.load_collection()
        return vdb
    except Exception as e:
        logger.error(f"Failed to create VDB client: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to {vdb_name}: {str(e)}")


@routes.get("/list_vdb")
async def list_vector_db() -> Dict[str, List[str]]:
    """List all supported vector database types."""
    logger.info("Listing supported vector databases")
    return {"supported_vector_dbs": list(DEFAULT_SEARCH_CONFIGS.keys())}


@routes.get("/list_collections")
async def list_vector_collections(
    vdb = Depends(get_vdb_client)
) -> Dict[str, List[str]]:
    """List all available vector collections."""
    collections = vdb.list_collections()
    logger.info(f"Successfully listed {len(collections)} collections for VDB: {vdb.__class__.__name__}")
    return {"collections": collections}


@routes.get("/collection_stats")
async def get_collection_stats(
    vdb = Depends(get_vdb_client)
) -> Dict[str, Any]:
    """Get statistics of a specific collection."""
    vdb.load_collection()
    stats = vdb.get_collection_stats()
    logger.info(f"Successfully retrieved stats for collection: {vdb.collection_name} in VDB: {vdb.__class__.__name__}")
    return {"collection_stats": stats}


@routes.delete("/delete_collection")
async def delete_collection(
    vdb = Depends(get_vdb_client)
) -> Dict[str, str]:
    """Delete a specific collection."""
    vdb.load_collection()
    vdb.drop_collection()
    logger.info(f"Successfully deleted collection: {vdb.collection_name} in VDB: {vdb.__class__.__name__}")
    return {"message": f"Collection '{vdb.collection_name}' deleted successfully."}
