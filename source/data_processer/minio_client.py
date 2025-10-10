import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env.prod")
from minio import Minio

MINIO_ENDPOINT = f"localhost:{os.getenv('MINIO_PORT')}"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

class MinioClient:
  def __init__(self,
               minio_endpoint: str,
               minio_access_key: str,
               minio_secret_key: str, 
               bucket_name: str):
    self.bucket_name = bucket_name 
    self.client = Minio(
      endpoint=minio_endpoint,
      access_key=minio_access_key,
      secret_key=minio_secret_key,
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
        self.client.remove_bucket(bucket_name=self.bucket_name)
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
  
  def get_categories(self) -> list[str]:
    """Get list of all categories"""
    categories = set()
    
    try:
      # List all objects in the bucket with 'images/' prefix
      objects = self.client.list_objects(
        bucket_name=self.bucket_name, 
        prefix="images/", 
        recursive=False
      )
      
      for obj in objects:
        if obj.is_dir:  
          category = obj.object_name.split("/")[-2] 
          categories.add(category)

      return categories
    
    except Exception as e: 
       print(f"Error get list categories: {str(e)}")
       pass

  def get_images_in_category(self, category: str) -> list[str]:
    """Get all images in a specific category"""
    try:
      objects = self.client.list_objects(
          bucket_name=self.bucket_name,
          prefix=f"images/{category}/",
          recursive=True
      )
      
      images = []
      for obj in objects:
          # Only add actual files, not folders
          if obj.object_name.endswith(('.jpg', '.jpeg', '.png')):
              images.append(obj.object_name)
        
      return sorted(images)
    except Exception as e:
      print(f"Error getting images for category {category}: {e}")
      return []

  def get_image_url(self, object_name: str, expires_in_seconds: int = 3600) -> str:
    """
    Generate presigned URL for an image
    
    Args:
        object_name: Path to the image in MinIO (e.g., 'images/A/img1.jpg')
        expires_in_seconds: URL expiration time in seconds (default 1 hour)
    
    Returns:
        str: Presigned URL for the image
    """
    try:
        from datetime import timedelta
        url = self.client.presigned_get_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            expires=timedelta(seconds=expires_in_seconds)
        )
        return url
    except Exception as e:
        print(f"Error generating URL for {object_name}: {e}")
        return ""
