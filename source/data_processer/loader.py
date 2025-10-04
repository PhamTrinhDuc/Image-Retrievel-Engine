import os
import sys
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.helpers import create_logger
from minio_client import MinioClient


logger = create_logger("data_loader")


class DataLoader:
    def __init__(self, bucket_name: str, source_folder: str, max_workers=8, batch_size=50):
      self.source_folder = source_folder
      self.max_workers = max_workers
      self.batch_size = batch_size
      self.client = MinioClient(bucket_name=bucket_name)
      self.client.create_bucket_if_not_exists()

    # ----------------------
    # File discovery
    # ----------------------
    def get_all_images(self, subfolders=None):
        """Get all images from given folder(s)"""
        if subfolders is None:
            subfolders = ["train", "val"]

        image_paths = []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

        for subfolder in subfolders:
            subfolder_path = os.path.join(self.source_folder, subfolder)
            if os.path.exists(subfolder_path):
                for ext in extensions:
                    # Lowercase
                    images = glob.glob(os.path.join(subfolder_path, "**", ext), recursive=True)
                    # Uppercase
                    images.extend(glob.glob(os.path.join(subfolder_path, "**", ext.upper()), recursive=True))
                    image_paths.extend(images)

                logger.info(f"Found {len([p for p in image_paths if subfolder in p])} images in {subfolder}")

        logger.info(f"Total images found: {len(image_paths)}")
        return image_paths

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
            logger.error(f"Error getting bucket stats: {e}")