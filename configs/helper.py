from dataclasses import dataclass
import os 


@dataclass
class DataConfig: 
  data_folder: str = "/mnt/c/Users/Admin/Downloads/archive/animals/animals"
  bucket_name: str = "animal-images"
  minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
  minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
  minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
  batch_size: int = 32
  num_workers: int = 8
  pin_memory: bool = True
  shuffle: bool = True
  valid_split: float = 0.2
  random_seed: int = 42

