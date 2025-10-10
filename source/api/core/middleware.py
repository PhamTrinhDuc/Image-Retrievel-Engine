import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from utils.helpers import trace_id_ctx

class TraceIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Tạo trace_id duy nhất cho mỗi request
        trace_id = str(uuid.uuid4())
        
        # Set context với token để đảm bảo context được truyền đúng
        token = trace_id_ctx.set(trace_id)
        
        try:
            response = await call_next(request)
            return response
        finally:
            # Reset context sau khi xử lý xong
            trace_id_ctx.reset(token)