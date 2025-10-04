from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

class BaseVectorDB(ABC): 
  """Abstract base class for vector database clients"""
  def __init__(self, host: str, port: str, collection_name: str):
    self.host = host
    self.port = port
    self.collection_name = collection_name
    self.collection = None

  @abstractmethod
  def connect(self) -> bool: 
    """Connect to vector database"""
    pass

  @abstractmethod
  def disconnect(self) -> bool: 
    """Disconnect from vector database"""
    pass

  @abstractmethod
  def create_collection(self, dim: int, description: str=""): 
    """Create a collection for storing embeddings"""
    pass

  @abstractmethod
  def load_collection(self): 
    """Load collection into memory"""
    pass

  @abstractmethod
  def insert_embeddings(self, image_paths: List[str], 
                        embeddings: np.ndarray, metadata: List[str]=None) -> List[int]: 
    """Insert embeddings into collection"""
    pass

  @abstractmethod
  def search_similar(self, query_embedding: np.ndarray, 
                     top_k: int=10, 
                     search_params: Dict=None) -> List[Dict]: 
    """Search for similar vectors"""
    pass

  @abstractmethod
  def delete_by_ids(self, ids: List[int]): 
    """Delete embeddings by IDs"""
    pass

  @abstractmethod
  def get_collection_stats(self) -> Dict[str, Any]: 
    """Get collection statistics"""
    pass 

  @abstractmethod
  def drop_collection(self): 
    """Drop the collection"""
    pass

  def __enter__(self):
    """Context manager entry"""
    self.connect()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit"""
    self.disconnect()