"""
pytorch lightning model to classify a sentence as positive or negative using the IMDB dataset from huggingface datasets
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import logging
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, learning_rate, weight_decay, train_batch_size, eval_batch_size, max_seq_length, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def prepare_data(self):
        dataset = load_dataset('imdb', split='train')
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        self.test_dataset = load_dataset('imdb', split='test')

    def collate_fn(self, batch):
        # collate using tokenizer
        # tokenize batch
        tokenized = self.tokenizer([x['text'] for x in batch], padding=True, truncation=True, max_length=self.max_seq_length)
        # convert to tensors
        labels = torch.tensor([x['label'] for x in batch])
        return {'input_ids': torch.tensor(tokenized['input_ids']), 'attention_mask': torch.tensor(tokenized['attention_mask']), 'label': labels}
        


    def train_dataloader(self):
        # create dataloader using the tokenizer
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=4, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=4, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=4, collate_fn=self.collate_fn)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro')
        return {'val_loss': loss, 'val_acc': torch.tensor(acc), 'val_precision': torch.tensor(precision), 'val_recall': torch.tensor(recall), 'val_f1': torch.tensor(f1)}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.log('val_precision', avg_precision)
        self.log('val_recall', avg_recall)
        self.log('val_f1', avg_f1)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'val_precision': avg_precision, 'val_recall': avg_recall, 'val_f1': avg_f1}
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro')
        return {'test_loss': loss, 'test_acc': torch.tensor(acc), 'test_precision': torch.tensor(precision), 'test_recall': torch.tensor(recall), 'test_f1': torch.tensor(f1)}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)
        self.log('test_acc', avg_acc)
        self.log('test_precision', avg_precision)
        self.log('test_recall', avg_recall)
        self.log('test_f1', avg_f1)
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'test_precision': avg_precision, 'test_recall': avg_recall, 'test_f1': avg_f1}
    
    def predict(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output.logits.detach().numpy()
        preds = np.argmax(scores, axis=1)
        return preds
    
# main function with argparse and training
def main(args):
    model = IMDBSentimentClassifier(args.model_name, args.num_labels, args.learning_rate, args.weight_decay, args.train_batch_size, args.eval_batch_size, args.max_seq_length)
    model.prepare_data()
    wandb_logger = WandbLogger(project='IMDB Sentiment Analysis', log_model=True, name=args.experiment_name)    
    #log args
    wandb_logger.log_hyperparams(args)
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs, accumulate_grad_batches=args.accumulate_grad_batches, logger=wandb_logger)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default='IMDB Sentiment Analysis')
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    main(args)
    #example of a shell command: python model.py --learning_rate 2e-5 --accumulate_grad_batches 3 --seed 42 --train_batch_size=16
    

