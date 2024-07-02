# Facebook Hateful Meme Dataset

This project contains a script to create a dataset folder containing memes and the corresponding `train.jsonl` file with corrected labels.

## Usage

To create the dataset folder containing the meme images and the corresponding `train.jsonl` file with corrected labels, run the following command:
```
python main.py
```
This will generate a fb-harmful-memes folder in the project directory with the following structure:
```
fb-harmful-memes/
│
├── memes/
│   ├── meme1.jpg
│   ├── meme2.jpg
│   └── ...
│
└── train.jsonl
```