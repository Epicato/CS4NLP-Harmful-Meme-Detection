import os
import json
import requests
from huggingface_hub import HfApi, Repository
from PIL import Image
from io import BytesIO

HF_USERNAME = "kyueran"
HF_REPO_NAME = "fb-hateful-memes"
HF_TOKEN = ""

# Paths and directories
jsonl_file_paths = ["./dev_seen.jsonl", "./dev_unseen.jsonl"]
images_dir = "img"

new_jsonl_file_path = "train.jsonl"

# Create the images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

# Function to download an image with error handling
def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return True
    except requests.exceptions.RequestException as e:
        return False

# Collect valid entries
valid_entries = []

# Process each JSONL file
for jsonl_file_path in jsonl_file_paths:
    with open(jsonl_file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    for entry in data:
        image_filename = entry["img"]
        image_url = f"https://huggingface.co/datasets/neuralcatcher/hateful_memes/resolve/main/{image_filename}"
        image_path = os.path.join(images_dir, os.path.basename(image_filename))
        
        if download_image(image_url, image_path):
            valid_entries.append(entry)

# Save valid entries to a new JSONL file
with open(new_jsonl_file_path, 'w') as file:
    for entry in valid_entries:
        file.write(json.dumps(entry) + '\n')

os.makedirs(f"./{HF_REPO_NAME}/images", exist_ok=True)
os.system(f"cp -r {images_dir}/* ./{HF_REPO_NAME}/images/")

os.system(f"cp {new_jsonl_file_path} ./{HF_REPO_NAME}/")

