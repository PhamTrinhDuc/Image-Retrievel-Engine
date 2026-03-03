import os
import shutil
from sklearn.model_selection import train_test_split
from source.configs.helper import DataConfig


def split_dataset(folder_dir, train_ratio=0.8, output_dir=None, max_images_per_class=1000):
  """
  Split dataset từ folder_dir thành train và val sets.
  
  Args:
    folder_dir: đường dẫn đến folder chứa 10 animal subfolders
    train_ratio: tỷ lệ train (mặc định 0.8 = 80% train, 20% val)
    output_dir: thư mục output (mặc định tạo train_val folder)
  """
  if output_dir is None:
    output_dir = os.path.join(os.path.dirname(folder_dir), "train_val")
  
  train_dir = os.path.join(output_dir, "train")
  val_dir = os.path.join(output_dir, "val")
  
  # Tạo thư mục output
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(val_dir, exist_ok=True)
  
  # Lặp qua từng animal folder
  for animal_class in os.listdir(folder_dir):
    class_path = os.path.join(folder_dir, animal_class)
    
    if not os.path.isdir(class_path):
      continue
    
    # Lấy danh sách tất cả ảnh
    images = os.listdir(class_path)[:max_images_per_class]
    
    # Split train/val
    train_images, val_images = train_test_split(
      images, 
      train_size=train_ratio, 
      random_state=42
    )
    
    # Tạo thư mục class trong train và val
    os.makedirs(os.path.join(train_dir, animal_class), exist_ok=True)
    os.makedirs(os.path.join(val_dir, animal_class), exist_ok=True)
    
    # Copy train images
    for img in train_images:
      src = os.path.join(class_path, img)
      dst = os.path.join(train_dir, animal_class, img)
      shutil.copy2(src, dst)
    
    # Copy val images
    for img in val_images:
      src = os.path.join(class_path, img)
      dst = os.path.join(val_dir, animal_class, img)
      shutil.copy2(src, dst)
    
    print(f"{animal_class}: {len(train_images)} train, {len(val_images)} val")


if __name__ == "__main__":
  folder_dir = DataConfig.raw_data
  output_dir = DataConfig.splitted_data
  split_dataset(folder_dir, train_ratio=0.8, output_dir=output_dir)