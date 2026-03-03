import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data_processer.loader import DataLoader
from configs.helper import DataConfig

SOURCE_FOLDER = os.path.join(DataConfig.splitted_data, "train")
BUCKET_NAME = DataConfig.bucket_name
MAX_WORKERS = DataConfig.num_workers
BATCH_SIZE = DataConfig.batch_size
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")  

def main():
  loader = DataLoader(bucket_name=BUCKET_NAME, 
                      source_folder=SOURCE_FOLDER,
                      minio_endpoint=MINIO_ENDPOINT,
                      minio_access_key=MINIO_ACCESS_KEY,
                      minio_secret_key=MINIO_SECRET_KEY,
                      max_workers=MAX_WORKERS, 
                      batch_size=BATCH_SIZE, 
                      size_limit=(150, 1500))  # (min, max) size in pixels

  loader.load_images_batch()     # Recommended
  # loader.load_images_parallel() 
  # loader.load_images_simple()

  loader.show_bucket_stats()

if __name__ == "__main__":
  main()

  # python -m source.operator.insert_images_to_minio
