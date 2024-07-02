import os
import json

# Path to the JSONL file
jsonl_file_path = "train.jsonl"
new_jsonl_file_path = "train_no_duplicates.jsonl"

# Function to remove duplicate image IDs
def remove_duplicate_image_ids(jsonl_file_path, new_jsonl_file_path):
    image_ids = set()
    unique_entries = []

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            image_filename = entry["img"]
            image_id, _ = os.path.splitext(os.path.basename(image_filename))
            
            if image_id not in image_ids:
                image_ids.add(image_id)
                unique_entries.append(entry)

    with open(new_jsonl_file_path, 'w') as file:
        for entry in unique_entries:
            file.write(json.dumps(entry) + '\n')

# Remove duplicate image IDs
remove_duplicate_image_ids(jsonl_file_path, new_jsonl_file_path)

print(f"Duplicates removed. Unique entries saved to {new_jsonl_file_path}.")
