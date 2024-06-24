import openai
from datasets import load_dataset
import time
from openai import OpenAI
import numpy as np
from tqdm import tqdm

from utils import classify_straight

# Load the dataset
# fb, harmC, harmP
dataset_type = "fb"
# the Base urls of the datasets are different
print("Loading dataset...")
if dataset_type == "fb":
    dataset = load_dataset("kyueran/fb-harmful-memes")
    base_url = "https://huggingface.co/datasets/kyueran/fb-harmful-memes/resolve/main/images/"
elif dataset_type == "harmC":
    dataset = load_dataset("kyueran/harm-c")
    base_url = "https://huggingface.co/datasets/kyueran/harm-c/resolve/main/images/covid_memes_"
elif dataset_type == "harmP":
    dataset = load_dataset("kyueran/harm-p")
    base_url = "https://huggingface.co/datasets/kyueran/harm-p/resolve/main/images/memes_"


# Process each meme in the dataset
gts = ["harmless", "harmful"]
acc = []
wrong_memes = []

for meme in tqdm(dataset['train']):

    if dataset_type == "fb":
        url = meme['img'].split('/')[-1]
    elif dataset_type == "harmC" or dataset_type == "harmP":
        url = meme['image'].split('_')[-1]
    
    meme_image_url = base_url + url
    # label is 0 or 1
    if dataset_type == "fb":
        ground_truth = gts[int(meme['label'])]
    elif dataset_type == "harmC" or dataset_type == "harmP":
        ground_truth = gts[int(meme['labels'])]
    label = classify_straight(meme_image_url, ground_truth, verbose=False)
    
    # calculate accuracy
    if ground_truth.lower() in label.lower():
        acc.append(1)
    else:
        acc.append(0)
        wrong_memes.append(url[:-4])


print("acc: ", np.mean(acc))

np.save('wrong_indices_' + dataset_type + '.npy', np.asarray(wrong_memes))