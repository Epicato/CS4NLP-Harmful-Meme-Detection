import os
import json
import shutil

# Paths and directories
jsonl_file_path = "harm-p-train.jsonl"
source_images_dir = "./harmeme_images_us_pol"
destination_images_dir = "./images"

# Create the destination images directory if it doesn't exist
os.makedirs(destination_images_dir, exist_ok=True)

# Function to copy images listed in the JSONL file to a new directory
def copy_images(jsonl_file_path, source_images_dir, destination_images_dir):
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            image_filename = entry["image"]
            source_image_path = os.path.join(source_images_dir, image_filename)
            destination_image_path = os.path.join(destination_images_dir, image_filename)
            
            if os.path.exists(source_image_path):
                shutil.copy2(source_image_path, destination_image_path)
                print(f"Copied {image_filename} to {destination_images_dir}")
            else:
                print(f"Image {image_filename} not found in source directory.")

# Copy the images
copy_images(jsonl_file_path, source_images_dir, destination_images_dir)

print(f"Images listed in {jsonl_file_path} have been copied to {destination_images_dir}.")
