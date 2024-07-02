import os
import json

# Paths and directories
jsonl_file_path = "train.jsonl"
images_dir = "img"

# Function to read image IDs from the JSONL file
def read_image_ids(jsonl_file_path):
    image_ids = set()
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            image_filename = entry["img"]
            image_id, _ = os.path.splitext(os.path.basename(image_filename))
            image_ids.add(image_id)
    return image_ids

# Function to list PNG files in the images directory
def list_png_files(images_dir):
    png_files = set()
    for file_name in os.listdir(images_dir):
        if file_name.endswith(".png"):
            image_id, _ = os.path.splitext(file_name)
            png_files.add(image_id)
    return png_files

# Read image IDs from the JSONL file
image_ids_in_jsonl = read_image_ids(jsonl_file_path)
print(len(image_ids_in_jsonl))

# List PNG files in the images directory
image_ids_in_img_dir = list_png_files(images_dir)

# Find image IDs that are in the JSONL file but not in the images directory
missing_image_ids = image_ids_in_jsonl - image_ids_in_img_dir

# Print the missing image IDs
print(f"Missing image IDs ({len(missing_image_ids)}):")
for image_id in missing_image_ids:
    print(image_id)
