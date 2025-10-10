#!/usr/bin/env python3
"""
Simple script to run the Image Retrieval API server
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv('../../.env.dev')
load_dotenv()

from fastapi import FastAPI
from starlette.responses import Response as HttpResponse
from api.routes.route_retriever import routes as retriever_router
from api.routes.route_vdb import routes as vdb_router
from api.routes.route_embedder import routes as embedder_router
from api.routes.route_operate import routes as operate_router
from api.core.middleware import TraceIDMiddleware
from utils.helpers import create_logger


app = FastAPI(title="Image Retrieval API", version="1.0")
app.add_middleware(TraceIDMiddleware)
logger = create_logger()

# Register routers
app.include_router(retriever_router, prefix="/retriever", tags=["retriever"])
app.include_router(vdb_router, prefix="/vdb", tags=["vdb"])
app.include_router(embedder_router, prefix="/embedder", tags=["embedder"])
app.include_router(operate_router, prefix="/operate", tags=["operate"])

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return HttpResponse(status_code=200, content="API backend is healthy")

# script: uvicorn main:app --host 0.0.0.0 --port 8001 --reload