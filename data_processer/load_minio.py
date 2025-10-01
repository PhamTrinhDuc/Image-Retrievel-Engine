
import os
import glob

folder_path = "/mnt/c/Users/Admin/Downloads/archive/ECOMMERCE_PRODUCT_IMAGES"
images_path = []

for sub_folder in os.listdir(folder_path): 
  path = os.path.join(folder_path, sub_folder)
  if sub_folder == "train" or sub_folder == "val": 
    image_list = glob.glob(path + "/**/*.jpeg", recursive=True )
    images_path.extend(image_list)

print(len(images_path))