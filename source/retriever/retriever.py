import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import time
from PIL import Image

from vector_db.vdb_factory import VectorDBFactory
from embedder.extractor_factory import EmbedderFactory
from utils.helpers import create_logger
from configs.helper import DEFAULT_EXTRACTOR_CONFIGS, DEFAULT_SEARCH_CONFIGS


class ImageRetriever:
    """
    Dedicated class for image retrieval and similarity search.
    Focuses on inference and search operations with flexible configuration.
    """
    
    # Default configurations for different extractors (optimized for inference)
    DEFAULT_EXTRACTOR_CONFIGS = DEFAULT_EXTRACTOR_CONFIGS
    # Default search configurations for different vector databases
    DEFAULT_SEARCH_CONFIGS = DEFAULT_SEARCH_CONFIGS
    
    def __init__(self,
                 extractor_type: str,
                 extractor_config: Optional[Dict[str, Any]] = None,
                 vdb_type: str = "milvus",
                 vdb_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Image Retriever.
        
        Args:
            extractor_type: Type of extractor ('resnet', 'vgg', 'vit', 'dinov2')
            extractor_config: Custom configuration for the extractor
            vdb_type: Type of vector database ('milvus', 'pinecone', 'weaviate')
            vdb_config: Custom configuration for the vector database
        """
        self.extractor_type = extractor_type.lower()
        self.vdb_type = vdb_type.lower()
        
        # Setup logging
        self.logger = create_logger(job_name="image_retriever")
        
        # Merge configurations with defaults
        self.extractor_config = self._merge_config(
            self.DEFAULT_EXTRACTOR_CONFIGS.get(self.extractor_type, {}),
            extractor_config or {}
        )
        
        self.vdb_config = self._merge_config(
            self.DEFAULT_SEARCH_CONFIGS.get(self.vdb_type, {}),
            vdb_config or {}
        )
        self.vdb_config['collection_name'] = self.extractor_config.get('collection_name', '')  
        self.extractor_config.pop('collection_name', None)  # Remove to avoid confusion

        
        # Initialize components
        self.extractor = None
        self.vdb_client = None
        
        # State tracking
        self.is_connected = False
        self.collection_loaded = False
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0
        }
        
        self.logger.info(f"ImageRetriever initialized - Extractor: {extractor_type}, VDB: {vdb_type}")
    
    def _merge_config(self, default_config: Dict, custom_config: Dict) -> Dict:
        """Merge custom config with default config."""
        merged = default_config.copy()
        merged.update(custom_config)
        return merged
    
    def initialize_extractor(self):
        """Initialize the feature extractor with configuration."""
        try:
            self.extractor = EmbedderFactory.create_extractor(
                self.extractor_type,
                **self.extractor_config
            )
            self.logger.info(f"Extractor {self.extractor_type} initialized for inference")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize extractor: {str(e)}")
            return False
    
    def initialize_vdb_client(self):
        """Initialize the vector database client with configuration."""
        try:
            # Extract connection config (without search params)
            connection_config = {k: v for k, v in self.vdb_config.items() 
                               if k != 'search_params'}
            
            self.vdb_client = VectorDBFactory.create_client(
                self.vdb_type,
                **connection_config
            )
            self.logger.info(f"VDB client {self.vdb_type} initialized for search")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VDB client: {str(e)}")
            return False
    
    def connect_and_load(self) -> bool:
        """
        Connect to vector database and load collection for search.
        
        Returns:
            True if setup successful, False otherwise
        """
        # Initialize components if not done
        if not self.extractor:
            if not self.initialize_extractor():
                return False
        
        if not self.vdb_client:
            if not self.initialize_vdb_client():
                return False
        
        # Connect to database
        try:
            self.is_connected = self.vdb_client.connect()
            if not self.is_connected:
                self.logger.error("Failed to connect to vector database")
                return False
            
            self.logger.info(f"Connected to {self.vdb_type} successfully")
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False
        
        # Load collection for search
        try:
            # Check if collection exists
            if self.vdb_type == 'milvus':
                from pymilvus import utility
                collection_name = self.vdb_config['collection_name']
                
                if not utility.has_collection(collection_name):
                    self.logger.error(f"Collection '{collection_name}' does not exist")
                    return False
            
            # Load collection
            self.vdb_client.load_collection()
            self.collection_loaded = True
            
            self.logger.info(f"Collection loaded successfully for search")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load collection: {str(e)}")
            return False
    
    def extract_query_embedding(self, query_input: Union[str, Path, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract embedding from query input.
        
        Args:
            query_input: Query image path, PIL Image, or numpy array
            
        Returns:
            Query embedding array or None if failed
        """
        if not self.extractor:
            self.logger.error("Extractor not initialized")
            return None
        
        try:
            start_time = time.time()
            
            if isinstance(query_input, np.ndarray):
                # If already an embedding
                if query_input.ndim == 1:
                    embedding = query_input
                else:
                    self.logger.error("Invalid embedding array dimensions")
                    return None
            else:
                # Extract embedding from image
                embedding = self.extractor.get_embedding(query_input)
            
            extraction_time = time.time() - start_time
            self.logger.debug(f"Query embedding extracted in {extraction_time:.3f}s")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to extract query embedding: {str(e)}")
            return None
    
    def search_similar_images(self,
                             query_input: Union[str, Path, Image.Image, np.ndarray],
                             top_k: int = 10,
    ):
        """
        Search for similar images in the database.
        
        Args:
            query_input: Query image path, PIL Image, numpy array, or embedding
            top_k: Number of similar images to return
            search_params: Custom search parameters for the vector database
            include_distances: Whether to include similarity distances
            filter_params: Additional filtering parameters
            
        Returns:
            List of similar images with metadata and scores
        """
        if not self.collection_loaded:
            self.logger.error("Collection not loaded. Call connect_and_load() first.")
            return []
        
        start_time = time.time()
        
        try:
            # Extract query embedding
            query_embedding = self.extract_query_embedding(query_input)
            if query_embedding is None:
                self.logger.error("Failed to get query embedding")
                return []
            
            
            # Perform search
            results = self.vdb_client.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            
            # Update performance stats
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            self.logger.info(f"Found {len(results)} similar images in {search_time:.3f}s")

            return results

        except Exception as e:
            self.logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def batch_search_similar_images(self,
                                   query_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
                                   top_k: int = 10,
                                   search_params: Optional[Dict] = None) -> List[List[Dict[str, Any]]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            query_inputs: List of query inputs
            top_k: Number of similar images to return per query
            search_params: Custom search parameters
            
        Returns:
            List of results for each query
        """
        if not query_inputs:
            return []
        
        batch_results = []
        
        for i, query_input in enumerate(query_inputs):
            try:
                results = self.search_similar_images(
                    query_input=query_input,
                    top_k=top_k,
                    search_params=search_params
                )
                batch_results.append(results)
                
                self.logger.debug(f"Processed query {i+1}/{len(query_inputs)}")
                
            except Exception as e:
                self.logger.error(f"Failed to process query {i+1}: {str(e)}")
                batch_results.append([])
        
        return batch_results
    
    def _update_search_stats(self, search_time: float):
        """Update search performance statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['total_search_time'] += search_time
        self.search_stats['avg_search_time'] = (
            self.search_stats['total_search_time'] / self.search_stats['total_searches']
        )
    
    def get_similar_by_id(self, 
                         image_id: int, 
                         top_k: int = 10,
                         exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        Get similar images by database ID.
        
        Args:
            image_id: ID of the image in the database
            top_k: Number of similar images to return
            exclude_self: Whether to exclude the query image from results
            
        Returns:
            List of similar images
        """
        if not self.collection_loaded:
            self.logger.error("Collection not loaded. Call connect_and_load() first.")
            return []
        
        try:
            # Get embedding by ID (this depends on VDB implementation)
            if hasattr(self.vdb_client, 'get_embedding_by_id'):
                query_embedding = self.vdb_client.get_embedding_by_id(image_id)
                if query_embedding is None:
                    self.logger.error(f"Image with ID {image_id} not found")
                    return []
                
                # Search for similar images
                results = self.search_similar_images(
                    query_input=query_embedding,
                    top_k=top_k + (1 if exclude_self else 0)
                )
                
                # Remove self if requested
                if exclude_self:
                    results = [r for r in results if r.get('id') != image_id][:top_k]
                
                return results
            else:
                self.logger.error("Vector database doesn't support get_embedding_by_id")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting similar images by ID: {str(e)}")
            return []
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get comprehensive retriever statistics and information."""
        return {
            "extractor_type": self.extractor_type,
            "extractor_config": self.extractor_config,
            "vdb_type": self.vdb_type,
            "vdb_config": self.vdb_config,
            "is_connected": self.is_connected,
            "collection_loaded": self.collection_loaded,
            "search_stats": self.search_stats.copy(),
            "extractor_info": self.extractor.get_model_info() if hasattr(self.extractor, 'get_model_info') else {}
        }
    
    def reset_stats(self):
        """Reset search statistics."""
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0
        }
        self.logger.info("Search statistics reset")
    
    def disconnect(self):
        """Disconnect from vector database and cleanup resources."""
        try:
            if self.is_connected and self.vdb_client:
                self.vdb_client.disconnect()
                self.is_connected = False
                self.collection_loaded = False
                
            self.logger.info(f"Disconnected from {self.vdb_type}")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.disconnect()


# Example usage
def example_retriever_usage():
    """Example of how to use the ImageRetriever."""
    
    # Custom configuration for DINOv2 extractor (optimized for inference)
    dinov2_config = {
        'model_name': 'dinov2_vitb14',
        'device': 'cpu',  # or 'cuda' if available
        'batch_size': 1,  # Single image inference
        'embed_dim': 768,
        'enable_mixed_precision': False
    }
    
    # Custom configuration for Milvus search
    milvus_config = {
        'host': 'localhost',
        'port': '19530',
        'collection_name': 'product_images',
        'search_params': {
            'metric_type': 'COSINE',
            'params': {'nprobe': 32}  # Higher for better accuracy
        }
    }
    
    # Initialize retriever
    retriever = ImageRetriever(
        extractor_type="dinov2",
        extractor_config=dinov2_config,
        vdb_type="milvus",
        vdb_config=milvus_config
    )
    
    try:
        # Connect and load collection
        if not retriever.connect_and_load():
            print("Failed to setup retriever")
            return
        
        # Search by image file
        results = retriever.search_similar_images(
            query_input="/path/to/query_image.jpg",
            top_k=10,
            include_distances=True
        )
        print(f"Found {len(results)} similar images")
        
        # Batch search
        query_images = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
        batch_results = retriever.batch_search_similar_images(
            query_inputs=query_images,
            top_k=5
        )
        print(f"Batch search completed: {len(batch_results)} queries processed")
        
        # Get statistics
        stats = retriever.get_retriever_stats()
        print(f"Retriever stats: {stats['search_stats']}")
        
    finally:
        retriever.disconnect()


if __name__ == "__main__":
    example_retriever_usage()
