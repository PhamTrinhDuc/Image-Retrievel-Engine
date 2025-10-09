#!/usr/bin/env python3
"""
Simple script to run the Image Retrieval API server
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
import uvicorn
from api.routes.route_retriever import routes as retriever_router
from api.routes.route_vdb import routes as vdb_router
from api.routes.route_embedder import routes as embedder_router
from api.routes.route_operate import routes as operate_router

app = FastAPI(title="Image Retrieval API", version="1.0")

# Register routers
app.include_router(retriever_router, prefix="/retriever", tags=["retriever"])
app.include_router(vdb_router, prefix="/vdb", tags=["vdb"])
app.include_router(embedder_router, prefix="/embedder", tags=["embedder"])
app.include_router(operate_router, prefix="/operate", tags=["operate"])

# script: uvicorn main:app --host 0.0.0.0 --port 8001 --reload