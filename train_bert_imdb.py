import torch
import datasets
import transformers
import torch.nn as nn
import os
import argparse
import numpy as np
import evaluate

from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

from transformers import Trainer, AutoModelForSequenceClassification,  TrainingArguments, BertTokenizer

def imdb_load():
    imdb_train = load_dataset('imdb', split='train', cache_dir='data')
    imdb_eval = load_dataset('imdb', split='test', cache_dir='data')
    
    return imdb_train, imdb_eval

def compute_metrics(eval_pred, metric: evaluate.Metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="data")
    train_dataset, eval_dataset = imdb_load()
    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2, torch_dtype="auto", cache_dir="data")

    training_args = TrainingArguments(output_dir=ckpt_dir, 
                                    eval_strategy="epoch",
                                    per_device_train_batch_size=args.batch_size,
                                    learning_rate=args.lr,
                                    num_train_epochs=args.epochs,
                                    save_strategy="epoch",
                                    tf32=True,)
    
    metric = evaluate.load("accuracy")
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,    
    eval_dataset=eval_dataset,
    compute_metrics=partial(compute_metrics, metric=metric),
    )
    
    trainer.train()

    