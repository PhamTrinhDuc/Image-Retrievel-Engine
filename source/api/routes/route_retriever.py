import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from models.base import SearchResponse, HealthResponse
from api.utils.helper import  get_retriever

from utils.helpers import create_logger

# Initialize logger
logger = create_logger("route_search")

# Initialize FastAPI app
routes = APIRouter()


# API Endpoints
@routes.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Try to get default retriever
        retriever = get_retriever(logger=logger, extractor_type="resnet")
        return HealthResponse(
            status="healthy",
            message="Image Retrieval API is running",
            code=200,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy", 
            message=f"Service initialization failed: {str(e)}",
            code=500
        )

@routes.post("/upload", response_model=SearchResponse)
async def search_by_upload(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    extractor_type: str = Form("resnet")
):
    """Search similar images by uploading an image file"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get retriever
        current_retriever = get_retriever(logger=logger, extractor_type=extractor_type)
        
        # Perform search
        import time
        start_time = time.time()
        
        results = current_retriever.search_similar_images(
            query_input=image,
            top_k=top_k
        )
        
        query_time = time.time() - start_time
        
        # Format results
        formatted_results = []
        for result in results:
          formatted_results.append({
              "image_id": result.get("id", "unknown"),
              "similarity_score": float(result.get("distance", 0.0)),
              "metadata": result.get("schema", {})
          })
        
        return SearchResponse(
            success=True,
            results=formatted_results,
            query_time=query_time,
            message=f"Found {len(results)} similar images"
        )
        
    except Exception as e:
        logger.error(f"Search by upload failed: {str(e)}")
        return SearchResponse(
            success=False,
            results=[],
            query_time=0.0,
            message=f"Search failed: {str(e)}"
        )

@routes.get("/random")
async def search_random_images(
    top_k: int = Query(5, ge=1, le=50),
    extractor_type: str = Query("resnet")
):
    """Get random images from the database (for testing)"""
    try:
        current_retriever = get_retriever(extractor_type)
        
        # Get random results (implement this in your retriever if needed)
        # For now, return a simple message
        return JSONResponse({
            "success": True,
            "message": f"Random search with {extractor_type} - feature not implemented yet",
            "top_k": top_k
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Random search failed: {str(e)}"
        })

