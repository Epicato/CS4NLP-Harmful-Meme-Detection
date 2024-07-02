import os
import json
from collections import defaultdict

# Path to the JSONL file
jsonl_file_path = "train.jsonl"

# Function to read image IDs from the JSONL file and identify duplicates
def find_duplicate_image_ids(jsonl_file_path):
    image_ids = defaultdict(int)
    duplicates = []

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            image_filename = entry["img"]
            image_id, _ = os.path.splitext(os.path.basename(image_filename))
            image_ids[image_id] += 1
            if image_ids[image_id] == 2:
                duplicates.append(image_id)

    return duplicates

# Find duplicate image IDs
duplicate_image_ids = find_duplicate_image_ids(jsonl_file_path)

# Print the duplicate image IDs
print(f"Duplicate image IDs ({len(duplicate_image_ids)}):")
for image_id in duplicate_image_ids:
    print(image_id)
