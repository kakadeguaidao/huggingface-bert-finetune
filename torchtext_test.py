import torch
import datasets

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import pipeline, BertTokenizer, BertModel, Trainer, TrainingArguments

def tokenizer_function(tokenizer, example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

if __name__ == "__main__":
    
    # load imby datasets

    imdb_train = load_dataset('imdb', cache_dir='data')
    # imdb_train = load_dataset('imdb', split='train', cache_dir='data')
    # imdb_eval = load_dataset('imdb', split='test', cache_dir='data')

    # imdb[0] {"text": str, "label": 0 or 1}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="data")
    # model = BertModel.from_pretrained("bert-base-uncased", cache_dir="data")
    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # print(output.shape)

    training_args = TrainingArguments(
        output_dir="results",
        per_device_train_batch_size=32,
        do_train=True,
        
    )