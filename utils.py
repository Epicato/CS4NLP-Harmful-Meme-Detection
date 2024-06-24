import openai
from datasets import load_dataset
import time
from openai import OpenAI
import numpy as np
from tqdm import tqdm

model_chat = "gpt-4o"
client = OpenAI()

def openai_request_with_retries(request_func, max_retries=3, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            response = request_func(**kwargs)
            return response
        except openai.APIError as e:
            print(f"API error: {e}. Retrying...")
            retries += 1
            time.sleep(2 ** retries)  # Exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    return None

def extract_text(meme_image_url):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the text in this meme and answer with only the text"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": meme_image_url,
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
        # max_tokens=150
    )
    if response:
        return response.choices[0].message.content.strip()
    return "Error extracting text"

def analyze_image(meme_image_url):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image excluding the text"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": meme_image_url,
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
        # max_tokens=150
    )
    if response:
        return response.choices[0].message.content.strip()
    return "Error analyzing image"

def analyze_context(meme_image_url):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the context of this meme"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": meme_image_url,
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
        # max_tokens=300
    )
    if response:
        return response.choices[0].message.content.strip()
    return "Error analyzing context"


def combine_and_reason(text_output, image_output, context_output):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {"role": "user", "content": (
                f"Given the following analyses of a meme, give the reasoning steps to determine if this meme is harmful or harmless. Make your answer brief\n"
                f"Text: {text_output}\nImage: {image_output}\nContext Analysis: {context_output}"
            )}
        ],
        # max_tokens=200
    )
    if response:
        return response.choices[0].message.content.strip()
    return "Error combining and reasoning"


def label_meme(reasoning_output):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {"role": "user", "content": (
                f"Based on the following reasoning process, classify the meme as harmful or harmless and answer with only one word: \"harmful\" or \"harmless\":\n"
                f"{reasoning_output}"
            )}
        ],
        max_tokens=50
    )
    if response:
        return response.choices[0].message.content.strip()
    return "Error labeling meme"

def refine_rationales(meme_image_url, reasoning_output, ground_truth):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {"role": "user", "content": (
                f"The previous analysis of a meme is: {reasoning_output}\n"
                f"The definition of harmfulness is: A direct or indirect attack on people based on characteristics, including ethnicity, \
                race, nationality, immigration status, religion, caste, sex, gender identity, sexual \
                orientation, and disability or disease. We define attack as violent or dehumanizing \
                (comparing people to non-human things, e.g. animals) speech, statements of \
                inferiority, and calls for exclusion or segregation. Mocking hate crime is also \
                considered harmful.\n"
                f"Given the definition above, refine the analysis of the meme and provide improved rationales for why the meme is harmful or harmless. Make your answer brief."
            )}
        ],
        # max_tokens=200
    )
    if response:
        return response.choices[0].message.content.strip()
    return "Error refining rationales"


# Main function to classify a meme
def classify_meme(meme_image_url, ground_truth, with_refine = False, verbose = False):

    # Step 1: Analyze text, image, and context separately using the meme as input
    text_output = extract_text(meme_image_url)
    image_output = analyze_image(meme_image_url)
    context_output = analyze_context(meme_image_url)
    
    # Step 2: Combine the outputs and give the rationales
    reasoning_output = combine_and_reason(text_output, image_output, context_output)
        
    # Step 3: Label the meme
    label_output = label_meme(reasoning_output)
    
    # Step 4: Refine the rationales
    if with_refine:
        refined_rationales = refine_rationales(meme_image_url, reasoning_output, ground_truth)
        
        # Repeat reasoning and labeling with refined rationales
        label_output_new = label_meme(refined_rationales)
    
    # print the outputs if needed (for debugging)
    if verbose:
        print("MEME: ",meme_image_url)
        print("TEXT: ",text_output)
        print("IMAGE: ",image_output)
        print("CONTEXT: ", context_output)
        print("RATIONALE: ", reasoning_output)
        print("PRED: ", label_output)
        if with_refine:
            print("REFINED_RATIONALE: ",refined_rationales)
            print("REFINED_PRED: ", label_output_new)
        print("GT: ", ground_truth)
    
    if with_refine:
        return label_output_new
    else:
        return label_output



def classify_straight(meme_image_url, ground_truth, verbose = False):
    response = openai_request_with_retries(
        client.chat.completions.create,
        model=model_chat,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this meme as harmful or harmless and answer with only one word: \"harmful\" or \"harmless\""},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": meme_image_url,
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
        max_tokens=50
    )
    if response:
        label = response.choices[0].message.content.strip()
    else:
        return "Error labeling meme"
        
    # print the outputs if needed (for debugging)
    if verbose:
        print("MEME: ",meme_image_url)
        print("PRED: ", label)
        print("GT: ", ground_truth)
    
    return label