import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from vector_db.vdb_factory import VectorDBFactory
from embedder.extractor_factory import EmbedderFactory
from utils.helpers import create_logger


class ImageEmbeddingLoader:
    """
    Dedicated class for loading images, extracting embeddings, and pushing to vector database.
    Focuses on data ingestion pipeline with flexible configuration.
    """
    
    # Default configurations for different extractors
    DEFAULT_EXTRACTOR_CONFIGS = {
        'resnet': {
            'model_name': 'resnet34',
            'device': 'cpu',
            'batch_size': 32,
            'enable_mixed_precision': False
        },
        'vgg': {
            'model_name': 'vgg16',
            'device': 'cpu',
            'batch_size': 32,
            'enable_mixed_precision': False
        },
        'vit': {
            'model_name': 'vit_base_patch16_224',
            'device': 'cpu',
            'batch_size': 16,
            'enable_mixed_precision': False
        },
        'dinov2': {
            'model_name': 'dinov2_vitb14',
            'device': 'cpu',
            'batch_size': 16,
            'enable_mixed_precision': False
        }
    }
    
    # Default configurations for different vector databases
    DEFAULT_VDB_CONFIGS = {
        'milvus': {
            'host': 'localhost',
            'port': '19530',
            'collection_name': 'image_embeddings',
            'metric_type': 'L2',
            'index_type': 'IVF_FLAT',
            'index_params': {'nlist': 128}
        },
        'pinecone': {
            'api_key': '',
            'environment': 'us-west1-gcp',
            'index_name': 'image-embeddings',
            'dimension': 768,
            'metric': 'cosine'
        },
        'weaviate': {
            'host': 'localhost',
            'port': 8080,
            'class_name': 'ImageEmbedding',
            'scheme': 'http'
        }
    }
    
    def __init__(self,
                 extractor_type: str,
                 extractor_config: Optional[Dict[str, Any]] = None,
                 vdb_type: str = "milvus",
                 vdb_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Image Embedding Loader.
        
        Args:
            extractor_type: Type of extractor ('resnet', 'vgg', 'vit', 'dinov2')
            extractor_config: Custom configuration for the extractor
            vdb_type: Type of vector database ('milvus', 'pinecone', 'weaviate')
            vdb_config: Custom configuration for the vector database
        """
        self.extractor_type = extractor_type.lower()
        self.vdb_type = vdb_type.lower()
        
        # Setup logging
        self.logger = create_logger(job_name="image_embedding_loader")
        
        # Merge configurations with defaults
        self.extractor_config = self._merge_config(
            default_config=self.DEFAULT_EXTRACTOR_CONFIGS.get(self.extractor_type, {}),
            custom_config=extractor_config or {}
        )
        
        self.vdb_config = self._merge_config(
            default_config=self.DEFAULT_VDB_CONFIGS.get(self.vdb_type, {}),
            custom_config=vdb_config or {}
        )
        
        # Initialize components
        self.extractor = None
        self.vdb_client = None
        
        # State tracking
        self.is_connected = False
        self.collection_ready = False
        
        self.logger.info(f"ImageEmbeddingLoader initialized - Extractor: {extractor_type}, VDB: {vdb_type}")
    
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
            self.logger.info(f"Extractor {self.extractor_type} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize extractor: {str(e)}")
            return False
    
    def initialize_vdb_client(self):
        """Initialize the vector database client with configuration."""
        try:
            self.vdb_client = VectorDBFactory.create_client(
                self.vdb_type,
                **self.vdb_config
            )
            self.logger.info(f"VDB client {self.vdb_type} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VDB client: {str(e)}")
            return False
    
    def connect_and_setup(self, recreate_collection: bool = False) -> bool:
        """
        Connect to vector database and setup collection.
        
        Args:
            recreate_collection: Whether to recreate collection if it exists
            
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
        
        # Setup collection
        try:
            # Handle existing collection
            if self.vdb_type == 'milvus':
                from pymilvus import utility
                collection_name = self.vdb_config['collection_name']
                
                if utility.has_collection(collection_name):
                    if recreate_collection:
                        utility.drop_collection(collection_name)
                        self.logger.info(f"Dropped existing collection: {collection_name}")
                    else:
                        self.vdb_client.load_collection()
                        self.collection_ready = True
                        self.logger.info(f"Using existing collection: {collection_name}")
                        return True
            
            # Create new collection
            feature_dim = self._get_feature_dimension()
            self.vdb_client.create_collection(
                dim=feature_dim,
                description=f"Image embeddings using {self.extractor_type} extractor"
            )
            
            self.vdb_client.load_collection()
            self.collection_ready = True
            
            self.logger.info(f"Collection setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup collection: {str(e)}")
            return False
    
    def _get_feature_dimension(self) -> int:
        """Get the feature dimension from the extractor."""
        return self.extractor.feature_dim.get('model_dim')
    
    def load_single_image(self, 
                         image_path: Union[str, Path],
                         metadata: str = "") -> Optional[int]:
        """
        Load a single image, extract embedding and push to vector database.
        
        Args:
            image_path: Path to the image file
            metadata: Optional metadata string
            
        Returns:
            Inserted ID if successful, None otherwise
        """
        if not self.collection_ready:
            self.logger.error("Collection not ready. Call connect_and_setup() first.")
            return None
        
        try:
            # Extract embedding
            embedding = self.extractor.get_embedding(str(image_path))
            
            if embedding is None or len(embedding) == 0:
                self.logger.error(f"Failed to extract embedding for {image_path}")
                return None
            
            # Insert into vector database
            ids = self.vdb_client.insert_embeddings(
                image_paths=[str(image_path)],
                embeddings=embedding.reshape(1, -1),
                metadata=[metadata]
            )
            
            if ids and len(ids) > 0:
                self.logger.debug(f"Inserted image {image_path} with ID {ids[0]}")
                return ids[0]
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
        
        return None
    
    def load_image_batch(self, 
                        image_paths: List[Union[str, Path]],
                        metadata_list: Optional[List[str]] = None) -> List[int]:
        """
        Load a batch of images, extract embeddings and push to vector database.
        
        Args:
            image_paths: List of image file paths
            metadata_list: Optional list of metadata strings
            
        Returns:
            List of inserted IDs
        """
        if not self.collection_ready:
            self.logger.error("Collection not ready. Call method connect_and_setup() first.")
            return []
        
        if not image_paths:
            self.logger.error("No image paths provided for batch loading")
            return []
        
        # Prepare metadata
        if metadata_list is None:
            metadata_list = [""] * len(image_paths)
        elif len(metadata_list) != len(image_paths):
            self.logger.warning("Metadata list length mismatch. Padding with empty strings.")
            metadata_list.extend([""] * (len(image_paths) - len(metadata_list)))
        
        try:
            # Extract embeddings in batch
            str_paths = [str(path) for path in image_paths]
            embeddings = self.extractor.get_batch_embeddings(str_paths)
            
            if embeddings is None or len(embeddings) == 0:
                self.logger.error("Failed to extract embeddings for batch")
                return []
            
            # Insert into vector database
            inserted_ids = self.vdb_client.insert_embeddings(
                image_paths=str_paths,
                embeddings=embeddings,
                metadata=metadata_list
            )
            
            if inserted_ids:
                self.logger.info(f"Successfully loaded {len(inserted_ids)} images")
                return inserted_ids
            
        except Exception as e:
            self.logger.error(f"Error loading image batch: {str(e)}")
        
        return []

    def get_loader_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loader configuration."""
        return {
            "extractor_type": self.extractor_type,
            "extractor_config": self.extractor_config,
            "vdb_type": self.vdb_type,
            "vdb_config": self.vdb_config,
            "is_connected": self.is_connected,
            "collection_ready": self.collection_ready,
            "feature_dimension": self._get_feature_dimension() if self.extractor else None
        }
    
    def disconnect(self):
      """Disconnect from vector database and cleanup resources."""
      try:
        if self.is_connected and self.vdb_client:
            self.vdb_client.disconnect()
            self.is_connected = False
            self.collection_ready = False
            
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
def example_loader_usage():
    """Example of how to use the ImageEmbeddingLoader."""
    
    # Custom configuration for DINOv2 extractor
    dinov2_config = {
        'model_name': 'dinov2_vitb14',
        'device': 'cpu',  # or 'cuda' if GPU available
        'batch_size': 8,
        'enable_mixed_precision': False
    }
    
    # Custom configuration for Milvus
    milvus_config = {
        'host': 'localhost',
        'port': '19530',
        'collection_name': 'product_images',
        'metric_type': 'COSINE',
        'index_type': 'IVF_SQ8',
        'index_params': {'nlist': 256}
    }
    
    # Initialize loader
    loader = ImageEmbeddingLoader(
        extractor_type="dinov2",
        extractor_config=dinov2_config,
        vdb_type="milvus",
        vdb_config=milvus_config
    )
    
    try:
        # Connect and setup
        if not loader.connect_and_setup(recreate_collection=False):
            print("Failed to setup loader")
            return
        
        # Load single image
        image_id = loader.load_single_image(
            image_path="/path/to/image.jpg",
            metadata="category:electronics, brand:apple"
        )
        print(f"Loaded image with ID: {image_id}")
        
        # Get loader information
        info = loader.get_loader_info()
        print(f"Loader info: {info}")
        
    finally:
        loader.disconnect()


if __name__ == "__main__":
    example_loader_usage()
