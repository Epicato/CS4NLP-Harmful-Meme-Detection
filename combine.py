import openai
from datasets import load_dataset
import time
from openai import OpenAI
import numpy as np
from tqdm import tqdm

from utils import classify_meme

# Load the dataset
# fb, harmC, harmP
dataset_type = "fb"
wrong_memes = np.load("wrong_indices_fb.npy")
# the Base urls of the datasets are different
print("Loading dataset...")
if dataset_type == "fb":
    dataset = load_dataset("kyueran/fb-harmful-memes")
    base_url = "https://huggingface.co/datasets/kyueran/fb-harmful-memes/resolve/main/images/"
elif dataset_type == "harmC":
    dataset = load_dataset("kyueran/harm-c")
    base_url = "https://huggingface.co/datasets/kyueran/harm-p/resolve/main/images/memes_"
elif dataset_type == "harmP":
    dataset = load_dataset("kyueran/harm-p")
    base_url = "https://huggingface.co/datasets/kyueran/harm-c/resolve/main/images/covid_memes_"


# Process each meme in the dataset
gts = ["harmless", "harmful"]
acc = []

hard_memes = []

for meme in tqdm(dataset['train']):
    

    if dataset_type == "fb":
        url = meme['img'].split('/')[-1]
    elif dataset_type == "harmC":
        url = meme['img'].split('_')[-1]
    elif dataset_type == "harmP":
        url = meme['img'].split('_')[-1]
        
    if url[:-4] not in wrong_memes:
        continue
    
    meme_image_url = base_url + url
    # label is 0 or 1
    ground_truth = gts[int(meme['label'])]
    
    label = classify_meme(meme_image_url, ground_truth, with_refine=False, verbose=False)
    
    # calculate accuracy
    if ground_truth.lower() in label.lower():
        acc.append(1)
    else:
        acc.append(0)
        hard_memes.append(url[:-4])

print("num of wrong memes: ", len(wrong_memes))
print("acc_combined: ", np.mean(acc))
print("new_acc: ", (len(dataset['train'])-len(wrong_memes)+len(wrong_memes)*np.mean(acc)) / len(dataset['train']))

np.save('hard_indices_' + dataset_type + '.npy', np.asarray(hard_memes))
