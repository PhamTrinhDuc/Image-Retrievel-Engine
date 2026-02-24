import os
import sys
import glob
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.helpers import create_logger
from data_processer.minio_client import MinioClient


logger = create_logger()


class DataLoader:
    def __init__(self, bucket_name: str, 
                 source_folder: str, 
                 minio_endpoint: str,
                 minio_access_key: str,
                 minio_secret_key: str,
                 max_workers: int, 
                 batch_size: int, 
                 size_limit: tuple):
      
      self.source_folder = source_folder
      self.max_workers = max_workers
      self.batch_size = batch_size
      self.client = MinioClient(bucket_name=bucket_name, 
                                minio_endpoint=minio_endpoint,
                                minio_access_key=minio_access_key,
                                minio_secret_key=minio_secret_key)
      self.size_limit = size_limit  # (min, max) size in pixels
      self.client.create_bucket_if_not_exists()

    # ----------------------
    # File discovery
    # ----------------------
    def get_all_images(self, max_folder: int=10) -> list[str]:
      list_animals = []
      animal_images = {}
      list_images = []

      for animal in os.listdir(self.source_folder)[:max_folder]: 
        path = os.path.join(self.source_folder, animal)
        animal_path = []
        for img in os.listdir(path):
          image_path = os.path.join(path, img)
          w, h = Image.open(image_path).size
          if h > self.size_limit[1] or w > self.size_limit[1]:
            continue
          if h < self.size_limit[0] or w < self.size_limit[0]:
            continue
          animal_path.append(image_path)

        # list_animals.append(animal)
        # animal_images[animal] = animal_path
        list_images.extend(animal_path)
      logger.info(f"Found {len(list_images)} images in {len(list_animals)} categories")

      return list_images

    # ----------------------
    # Upload helpers
    # ----------------------
    def _upload_batch(self, image_batch):
        results = []
        for img_path in image_batch:
            results.append(self.client.upload_image(img_path))
        return results

    def _log_results(self, images, success_count, fail_count, start_time):
        duration = time.time() - start_time
        success_rate = success_count / len(images) * 100 if images else 0

        logger.info("✅ Upload completed!")
        logger.info(f"📊 Total: {len(images)}, Success: {success_count}, Failed: {fail_count}")
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        logger.info(f"⏱️ Duration: {duration:.1f}s")
        if duration > 0:
            logger.info(f"🚀 Speed: {len(images)/duration:.1f} files/sec")

    # ----------------------
    # Upload strategies
    # ----------------------
    def load_images_parallel(self):
        """Parallel upload (each image submitted as a task)"""
        images = self.get_all_images()
        if not images:
            logger.warning("No images found!")
            return

        success_count, fail_count = 0, 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.client.upload_image, img) for img in images]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
                result = future.result()
                if result[1]:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(f"Failed {result[0]}: {result[2]}")

        self._log_results(images, success_count, fail_count, start_time)

    def load_images_batch(self):
        """Batch upload (each thread handles a batch)"""
        images = self.get_all_images()
        if not images:
            logger.warning("No images found!")
            return

        batches = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        logger.info(f"Processing {len(batches)} batches of size {self.batch_size}")

        success_count, fail_count = 0, 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._upload_batch, batch) for batch in batches]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                for result in future.result():
                    if result[1]:
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.error(f"Failed {result[0]}: {result[2]}")

        self._log_results(images, success_count, fail_count, start_time)

    def load_images_simple(self):
        """Sequential upload (for debugging)"""
        images = self.get_all_images()
        if not images:
            logger.warning("No images found!")
            return

        success_count, fail_count = 0, 0
        start_time = time.time()

        for img_path in tqdm(images, desc="Uploading"):
            result = self.client.upload_image(img_path)
            if result[1]:
                success_count += 1
            else:
                fail_count += 1
                logger.error(f"Failed {result[0]}: {result[2]}")

        self._log_results(images, success_count, fail_count, start_time)

    # ----------------------
    # Stats
    # ----------------------
    def show_bucket_stats(self):
        try:
            categories = self.client.get_categories()
            logger.info(f"📁 Categories found: {len(categories)}")
            for category in categories:
                images = self.client.get_images_in_category(category)
                logger.info(f"  📂 {category}: {len(images)} images")
        except Exception as e:
            raise ValueError("Error getting bucket stats") from e
            logger.error(f"Error getting bucket stats: {e}")