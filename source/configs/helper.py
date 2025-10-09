from dataclasses import dataclass
import os 


@dataclass
class DataConfig: 
  data_folder: str = "/mnt/c/Users/Admin/Downloads/archive/animals/animals"
  bucket_name: str = "animal-images"
  minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
  minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
  minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
  batch_size: int = 64
  num_workers: int = 8
  pin_memory: bool = True
  shuffle: bool = True
  valid_split: float = 0.2
  random_seed: int = 42


# Default configurations for different extractors (optimized for inference)
DEFAULT_EXTRACTOR_CONFIGS = {
  'resnet': {
      'model_name': 'resnet34',
      'collection_name': 'resnet_embedding',
      'device': 'cpu',
      'batch_size': 32,  # Smaller batch for inference
      'enable_mixed_precision': False
  },
  'vgg': {
      'model_name': 'vgg16',
      'collection_name': 'vgg_embedding',
      'device': 'cpu',
      'batch_size': 32,
      'enable_mixed_precision': False
  },
  'vit': {
      'model_name': 'vit_base_patch16_224',
      'collection_name': 'vit_embedding',
      'device': 'cpu',
      'batch_size': 32,
      'enable_mixed_precision': False
  },
  'dinov2': {
      'model_name': 'dinov2_vitb14',
      'collection_name': 'dinov2_embedding',
      'device': 'cpu',
      'batch_size': 32,
      'enable_mixed_precision': False
  }
}

# Default search configurations for different vector databases
DEFAULT_SEARCH_CONFIGS = {
  'milvus': {
      'host': 'localhost',
      'port': '19530',
      'collection_name': 'image_embedding',
      'search_params': {
          'metric_type': 'L2',
          'params': {'nprobe': 16}
      }
  },
#   'pinecone': {
#       'api_key': '',
#       'environment': 'us-west1-gcp',
#       'index_name': 'image-embeddings',
#       'search_params': {
#           'top_k': 10,
#           'include_metadata': True
#       }
#   },
#   'weaviate': {
#       'host': 'localhost',
#       'port': 8080,
#       'class_name': 'ImageEmbedding',
#       'scheme': 'http',
#       'search_params': {
#           'limit': 10,
#           'certainty': 0.7
#       }
#   }
}
