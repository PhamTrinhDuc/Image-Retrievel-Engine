import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from base.base_vdb import BaseVectorDB
from .milvus_client import MilvusClient

from dotenv import load_dotenv
load_dotenv('/home/ducpham/workspace/Image-Retrieval-Engine/.env.dev')
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

class VectorDBFactory:
    """Factory class to create vector database clients"""
    
    @staticmethod
    def create_client(vdb_name: str, **kwargs) -> BaseVectorDB:
        """
        Create a vector database client
        
        Args:
            vdb_name: Type of vector database ('milvus', 'pinecone', 'weaviate', etc.)
            **kwargs: Configuration parameters for the specific client
            
        Returns:
            Vector database client instance
        """
        
        if vdb_name == "milvus":
            return MilvusClient(
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                collection_name=kwargs.get("collection_name")
            )
        # elif vdb_name.lower() == "pinecone":
        #     return PineconeClient(
        #         api_key=kwargs.get("api_key"),
        #         environment=kwargs.get("environment"),
        #         collection_name=kwargs.get("collection_name", "image-embeddings")
        #     )
        else:
            raise ValueError(f"Unsupported vector database type: {vdb_name}")