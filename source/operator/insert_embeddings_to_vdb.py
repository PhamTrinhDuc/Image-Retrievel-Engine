from collections import defaultdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data_processer.minio_client import MinioClient
from retriever.loader import ImageEmbeddingLoader
from configs.helper import DataConfig
import argparse

BUCKET_NAME = DataConfig.bucket_name
MINIO_ENDPOINT = DataConfig.minio_endpoint
MINIO_ACCESS_KEY = DataConfig.minio_access_key
MINIO_SECRET_KEY = DataConfig.minio_secret_key

def get_images_from_minio() -> dict[str, list[str]]: 
  minio_client = MinioClient(bucket_name=BUCKET_NAME,
                              minio_endpoint=MINIO_ENDPOINT,
                              minio_access_key=MINIO_ACCESS_KEY,
                              minio_secret_key=MINIO_SECRET_KEY)
  categories = minio_client.get_categories()
  image_urls = defaultdict(list)
  for category_id, category in enumerate(categories, start=1): 
    images_object = minio_client.get_images_in_category(category=category)
    image_url = [minio_client.get_image_url(object_name=object_name, 
                                            expires_in_seconds=3600*10) for object_name in images_object]
    image_urls[category] = image_url
    # if category_id == 3: # For testing, remove this line to process all categories
    #   break 
  
  return image_urls


def load_embeddings_to_vdb(model: str, config_model: dict, 
                            vdb: str, vdb_config: dict, 
                           image_urls: list[str], metadata: list[str], 
                           recreate_collection: bool = False):
    
    loader = ImageEmbeddingLoader(extractor_type=model, 
                                  extractor_config=config_model,
                                  vdb_type=vdb, 
                                  vdb_config=vdb_config)
    
    if recreate_collection:
      loader.vdb_client.drop_collection()

    loader.connect_and_setup()
    loader.load_image_batch(image_paths=image_urls, 
                            metadata_list=metadata)

    loader.disconnect()


def run(model: str, 
        vdb: str, 
        collection_name: str, 
        recreate_collection: bool=False
        ):
  image_urls = get_images_from_minio()
  image_list = []
  metadata_list = []
  for category, urls in image_urls.items(): 
    image_list.extend(urls)
    metadata_list.extend([category] * len(urls))
    # break # For testing, remove this line to process all categories
  try: 
    load_embeddings_to_vdb(
      image_urls=image_list, 
      metadata=metadata_list,
      model=model,
      config_model=None,
      vdb=vdb,
      vdb_config={"collection_name": collection_name},
      recreate_collection=recreate_collection
    )
  except Exception as e:
    print(f"Error during loading embeddings to VDB: {str(e)}")
    raise Exception(f"Error during loading embeddings to VDB: {str(e)}")
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Insert image embeddings to vector database.")
  parser.add_argument("--model", type=str, default="resnet", help="Model type for embedding extraction.")
  parser.add_argument("--vdb", type=str, default="milvus", help="Vector database type.")
  parser.add_argument("--collection_name", type=str, default="images", help="Collection name in vector database.")
  parser.add_argument("--recreate_collection", action="store_true", help="Recreate collection in vector database.")

  args = parser.parse_args()

  run(model=args.model,
    vdb=args.vdb,
    collection_name=args.collection_name,
    recreate_collection=args.recreate_collection
  )
