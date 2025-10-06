from collections import defaultdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data_processer.minio_client import MinioClient
from retriever.loader import ImageEmbeddingLoader
import argparse

BUCKET_NAME = "ecommerce-product-images"

def get_images_from_minio() -> dict[str, list[str]]: 
  minio_client = MinioClient(bucket_name=BUCKET_NAME)
  categories = minio_client.get_categories()
  image_urls = defaultdict(list)
  for category in categories: 
    images_object = minio_client.get_images_in_category(category=category)
    image_url = [minio_client.get_image_url(object_name=object_name, 
                                            expires_in_seconds=3600*10) for object_name in images_object]
    image_urls[category] = image_url
  
  return image_urls


def load_embeddings_to_vdb(model: str, vdb: str, 
                           image_urls: list[str], metadata: list[str], 
                           recreate_collection: bool = False):
    
    loader = ImageEmbeddingLoader(extractor_type=model, 
                                  vdb_type=vdb)
    if recreate_collection:
      loader.vdb_client.drop_collection()

    loader.connect_and_setup()
    loader.load_image_batch(image_paths=image_urls, 
                            metadata_list=metadata)

    loader.disconnect()


def main_with_args(args):
  image_urls = get_images_from_minio()
  image_list = []
  metadata_list = []
  for category, urls in image_urls.items(): 
    image_list.extend(urls)
    metadata_list.extend([category] * len(urls))
    # break # For testing, remove this line to process all categories

  load_embeddings_to_vdb(
    image_urls=image_list, 
    metadata=metadata_list,
    model=args.model,
    vdb=args.vdb,
    recreate_collection=args.recreate_collection
  )
  
if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="Insert image embeddings to vector database.")
  parser.add_argument("--model", type=str, default="resnet", help="Model type for embedding extraction.")
  parser.add_argument("--vdb", type=str, default="milvus", help="Vector database type.")
  parser.add_argument("--recreate_collection", action="store_true", help="Recreate collection in vector database.")

  args = parser.parse_args()

  main_with_args(args=args)