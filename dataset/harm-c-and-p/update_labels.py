import json

# Path to the JSONL file
jsonl_file_path = "harm-p-train.jsonl"
new_jsonl_file_path = "train_updated.jsonl"

# Function to transform labels
def transform_labels(labels):
    if "not harmful" in labels:
        return 0
    elif "somewhat harmful" in labels or "very harmful" in labels:
        return 1
    return None

# Read and update the JSONL file
def update_labels_in_jsonl(jsonl_file_path, new_jsonl_file_path):
    updated_entries = []

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            transformed_label = transform_labels(entry["labels"])
            if transformed_label is not None:
                entry["labels"] = transformed_label
                updated_entries.append(entry)
            else:
                print(f"Invalid labels: {entry['id']}")

    
    with open(new_jsonl_file_path, 'w') as file:
        for entry in updated_entries:
            file.write(json.dumps(entry) + '\n')

# Update labels in the JSONL file
update_labels_in_jsonl(jsonl_file_path, new_jsonl_file_path)

print(f"Labels updated and saved to {new_jsonl_file_path}.")
