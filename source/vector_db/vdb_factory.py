import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from base.base_vdb import BaseVectorDB
from .milvus_client import MilvusClient

class VectorDBFactory:
    """Factory class to create vector database clients"""
    
    @staticmethod
    def create_client(db_type: str, **kwargs) -> BaseVectorDB:
        """
        Create a vector database client
        
        Args:
            db_type: Type of vector database ('milvus', 'pinecone', 'weaviate', etc.)
            **kwargs: Configuration parameters for the specific client
            
        Returns:
            Vector database client instance
        """
        if db_type.lower() == "milvus":
            return MilvusClient(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", "19530"),
                collection_name=kwargs.get("collection_name", "image_embeddings")
            )
        # elif db_type.lower() == "pinecone":
        #     return PineconeClient(
        #         api_key=kwargs.get("api_key"),
        #         environment=kwargs.get("environment"),
        #         collection_name=kwargs.get("collection_name", "image-embeddings")
        #     )
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")