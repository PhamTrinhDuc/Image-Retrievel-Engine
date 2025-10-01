import os
from dotenv import load_dotenv
load_dotenv()
from minio import Minio

MINIO_ENDPOINT = f"localhost:{os.getenv('MINIO_PORT')}"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

class MinioClient:
  def __init__(self, bucket_name: str):
    self.bucket_name = bucket_name 
    self.client = Minio(
      endpoint=MINIO_ENDPOINT,
      access_key=MINIO_ACCESS_KEY,
      secret_key=MINIO_SECRET_KEY,
      secure=False
    )

  def create_bucket_if_not_exists(self):
    if not self.client.bucket_exists(self.bucket_name):
        self.client.make_bucket(self.bucket_name)
        print(f"Created bucket: {self.bucket_name}")
    else:
        print(f"Bucket {self.bucket_name} already exists")

  def delete_bucket_if_not_exists(self):
    if self.client.bucket_exists(self.bucket_name):
        self.client.remove_bucket(self.bucket_name)
        print(f"Deleted bucket: {self.bucket_name}")
    else:
        print(f"Bucket {self.bucket_name} not already exists")
  
  def upload_image(self, image_path: str):
    """Upload image to MinIO with category structure: images/category/filename"""
    # Extract category from path (assumes structure: .../train_or_val/category/image.jpg)
    path_parts = image_path.split(os.sep)
    
    # Find category (folder before the image file)
    filename = os.path.basename(image_path)
    category = path_parts[-2]  # Get the parent folder name (category)
    
    # Create object name: images/category/filename
    object_name = f"images/{category}/{filename}"
    
    try:
        # Use fput_object to upload file from local path
        self.client.fput_object(
            bucket_name=self.bucket_name, 
            object_name=object_name, 
            file_path=image_path,
            content_type="image/jpeg"  # Set proper content type
        )
        return (image_path, True, object_name)
    except Exception as e:
        return (image_path, False, str(e))
  
  def upload_image_simple(self, image_path: str):
    """Upload image with just filename (no folder structure)"""
    filename = os.path.basename(image_path)
    object_name = f"images/{filename}"
    
    try:
        self.client.fput_object(
            bucket_name=self.bucket_name, 
            object_name=object_name, 
            file_path=image_path,
            content_type="image/jpeg"
        )
        return (image_path, True, object_name)
    except Exception as e:
        return (image_path, False, str(e))