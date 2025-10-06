from PIL import Image
import io
import base64
from fastapi import HTTPException
from retriever.retriever import ImageRetriever


# Utility functions
def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if "data:image" in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
    

def get_retriever(logger, extractor_type: str = "resnet") -> ImageRetriever:
  """Get or create retriever instance"""
  
  # Simple caching - create new retriever if type changes
  try:
    retriever = ImageRetriever(
        extractor_type=extractor_type,
        vdb_type="milvus"
    )
    retriever.connect_and_load()
    
    logger.info(f"Created new retriever with {extractor_type} extractor")
  except Exception as e:
    logger.error(f"Failed to create retriever: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to initialize retriever: {str(e)}")
  
  return retriever