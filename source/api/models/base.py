from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Request/Response models
class SearchRequest(BaseModel):
    image_base64: str
    top_k: Optional[int] = 5
    extractor_type: Optional[str] = "resnet"

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    query_time: float
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    code: int
    message: str