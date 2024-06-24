# Harmful Meme Detection and Reasoning

This is a course project of "Computational Semantics for NLP 2024" at ETH Zurich. Leveraging chain(tree)-of-thought prompting on GPT-4o, we have built a framework that helps detect harmful memes and provide step-by-step reasons. We've also explored how to refine the reasons given the definition of the label(harmful), and thus improve the accuracy of meme detection and provide a more transparent reasoning process.

## Dataset
The dataset we used is from the [facebook challenge](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/). We constructed part of the data to be our dataset in this project. It can be accessed [here](https://huggingface.co/datasets/kyueran/fb-harmful-memes). The dataset contains 512 memes with 198 harmful ones. You don't need to download the dataset for running the tests since it can be accessed through the dataset library of Hugging Face.

## Environment

We recommend a virtual environment with Python 3.9.  
You need to install `openai 1.25.0` and `transformers 4.38.2`
You also need an OpenAI API key. To set up the key, you can refer to the official [tutorial](https://platform.openai.com/docs/quickstart).

## Reproduce the experiments
First, run 
```
python straight.py
```
This code simply asks GPT to classify the memes as harmful or harmless without providing reasons.  
We see this as our baseline framework. The accuracy is 0.6172  
After running this you will get a file `wrong_indices_fb.npy`, containing the indices of wrongly labeled memes in this run. In our test there were 196 out of 512 memes . 

Next, run
```
python combine.py
```
This code uses the combined framework and provides the reasons for labeling.  
Note that we only run this on the wrongly labeled memes in the previous run, since we have a limited budget for the API.  
The accuracy is 0.7246  
After running this you will get a file `hard_indices_fb.npy`, containing the indices of wrongly labeled memes in this run with our combined framework. They are seen as hard to determine by the model. In our test there were 141 out of 196 memes.  

Finally, run
```
python refine.py
```
Based on the previous framework, we give the definition of the label to refine the reasoning steps of the model and ask it to give an improved answer.  
Note that we also only run this on the wrongly labeled memes in the previous run, since we have a limited budget for the API.  
The accuracy is 0.7559  
After running this you will get a file `failed_indices_fb.npy`, containing the indices of wrongly labeled memes in this run with refinement. The model failed to label them even with refinement. In our test there were 125 out of 141 memes.

## For your own tests
You can also try different datasets. We also provide the choices of harmC and harmP [dataset](https://aclanthology.org/2021.findings-emnlp.379/).
We didn't test on them since we have a limited budget.  
You will need to change the argument `dataset_type` to change the dataset.  
If you want to change the framework of the model, you can change the `classify_meme` function in `utils.py`  
If you need to check the intermediate outputs from the model, you can set the argument `verbose` to `True`
