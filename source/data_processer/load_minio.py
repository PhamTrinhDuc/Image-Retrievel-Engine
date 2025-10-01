import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from minio_client import MinioClient

FOLDER_PATH = "/mnt/c/Users/Admin/Downloads/archive/ECOMMERCE_PRODUCT_IMAGES"
BUCKET_NAME = "ecommerce-product-images"
MAX_WORKERS = 8  # Tăng số workers để upload nhanh hơn

def get_images_path() -> list[str]:
    """Get all JPEG images from train and val folders with category structure"""
    images_path = []
    for sub_folder in os.listdir(FOLDER_PATH):
        path = os.path.join(FOLDER_PATH, sub_folder)
        if sub_folder == "train" or sub_folder == "val":
            # Get all images from train/val/*/**.jpeg
            image_list = glob.glob(path + "/**/*.jpeg", recursive=True)
            images_path.extend(image_list)
    
    print(f"Found {len(images_path)} images to upload")
    
    return images_path

def upload_batch(client, image_batch):
    """Upload một batch ảnh"""
    results = []
    for img_path in image_batch:
        result = client.upload_image(img_path)
        results.append(result)
    return results

def load_to_minio_parallel():
    client = MinioClient(bucket_name=BUCKET_NAME)
    client.create_bucket_if_not_exists()
    images_path = get_images_path()
    print(f"Total images: {len(images_path)}")

    success_count = 0
    fail_count = 0

    # Sử dụng tqdm để hiển thị progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for img_path in images_path:
            future = executor.submit(client.upload_image, img_path)
            futures.append(future)
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            result = future.result()
            if result[1]:  # Success
                success_count += 1
                if success_count % 100 == 0:  # Log every 100 successful uploads
                    print(f"✅ Uploaded {success_count} images successfully")
            else:  # Failed
                fail_count += 1
                print(f"❌ Failed: {result[0]}, Error: {result[2]}")

    print(f"\n🎉 Upload finished: {success_count} success, {fail_count} fail")
    print(f"Success rate: {success_count/(success_count + fail_count)*100:.2f}%")

def load_to_minio_batch():
    """Alternative: Upload theo batch để giảm overhead"""
    client = MinioClient(bucket_name=BUCKET_NAME)
    client.create_bucket_if_not_exists()
    images_path = get_images_path()
    
    BATCH_SIZE = 50  # Upload 50 ảnh mỗi batch
    batches = [images_path[i:i+BATCH_SIZE] for i in range(0, len(images_path), BATCH_SIZE)]
    
    print(f"Total images: {len(images_path)}, Batches: {len(batches)}")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(upload_batch, client, batch) for batch in batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            for result in batch_results:
                if result[1]:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"❌ Failed: {result[0]}, Error: {result[2]}")
    
    print(f"\n🎉 Upload finished: {success_count} success, {fail_count} fail")

if __name__ == "__main__": 
    # Chọn một trong hai cách:
    # load_to_minio_parallel()  # Upload từng ảnh song song
    load_to_minio_batch()   # Upload theo batch