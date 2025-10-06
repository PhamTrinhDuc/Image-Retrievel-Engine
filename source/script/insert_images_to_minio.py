import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data_processer.loader import DataLoader

SOURCE_FOLDER = "/mnt/c/Users/Admin/Downloads/archive/ECOMMERCE_PRODUCT_IMAGES"
BUCKET_NAME = "ecommerce-product-images"
MAX_WORKERS = 8
BATCH_SIZE = 50

def main():
  loader = DataLoader(bucket_name=BUCKET_NAME, source_folder=SOURCE_FOLDER,
                      max_workers=MAX_WORKERS, batch_size=BATCH_SIZE)

  loader.load_images_batch()     # Recommended
  # loader.load_images_parallel() 
  # loader.load_images_simple()

  loader.show_bucket_stats()

if __name__ == "__main__":
    main()
