import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchtext
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
import gc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import nltk
from nltk.corpus import wordnet
import random

#nltk.download('wordnet')
#nltk.download('omw-1.4')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('text.csv')
weights = [len(df)/len(df[df['label'] == i]) for i in range(6)]
weights = torch.tensor(weights)
weights = weights.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

vocab_size = tokenizer.vocab_size
context_size = 64

class EmotionDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.context_size = context_size
        self.augment = False

    def __len__(self):
        return self.size
    
    def synonym_replacement(self, sentence, n):
        new_sentence = sentence.split()
        words = [word for word in new_sentence if word.isalpha()]
        random_words = random.sample(words, min(n, len(words)))
        for word in random_words:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                    synonyms.add(synonym)
            if synonyms:
                synonym = random.choice(list(synonyms))
                new_sentence = [synonym if w == word else w for w in new_sentence]
        return ' '.join(new_sentence)

    def __getitem__(self, idx):
        line = df.iloc[idx]
        label = line['label']
        text = line['text']
        if self.augment:
            text = self.synonym_replacement(text, top_n=10)
        encoded_dict = tokenizer.encode_plus(
                        text,                      
                        add_special_tokens = True, 
                        max_length = context_size,           
                        padding='max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
        tokens = encoded_dict['input_ids']
        tokens = tokens.flatten()
        attn_mask = encoded_dict['attention_mask']
        attn_mask = attn_mask.flatten()
        if tokens.shape[0] > context_size:
            tokens[context_size - 1] = 102
            tokens = tokens[:context_size]
            attn_mask = attn_mask[:context_size]
        return tokens, attn_mask, label
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

dataset_size = 210000
dataset = EmotionDataset(dataset_size)
val_size = 10000
test_size = 100000
train_size = dataset_size - val_size - test_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_set.augment = True

batch_size = 128
train_dataloader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g)
val_dataloader = DataLoader(val_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=os.cpu_count(),
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=g)
test_dataloader = DataLoader(test_set,
                             batch_size=64,
                             shuffle=False,
                             num_workers=os.cpu_count(),
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=g)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased",
                                          output_attentions = False,
                                          output_hidden_states = False)
        layers = [
                  nn.Linear(768, 6)]
        layer_modules = nn.ModuleList(layers)
        self.net = nn.Sequential(*layer_modules)
        
    def forward(self, x, m):
        output = self.bert(x, m).last_hidden_state
        output = output[:, 0, :].view(output.shape[0], -1)
        output = self.net(output)
        return output
    
class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupCosineSchedule, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.last_epoch / max(1, self.warmup_steps) * base_lr for base_lr in self.base_lrs]
        return [0.5 * (1.0 + np.cos(np.pi * (self.last_epoch - self.warmup_steps) / 
                                    (self.t_total - self.warmup_steps))) * base_lr for base_lr in self.base_lrs]
    
epochs = 4
total_steps = len(train_dataloader) * epochs
warmup_steps = 100
gc.collect()
torch.cuda.empty_cache()
model = Model()
model = nn.DataParallel(model, device_ids=[0])
model.to(device)
loss = nn.CrossEntropyLoss(weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0005)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
epoch = 0
best_val_acc = 0

@torch.no_grad()
def estimate_correct(prediction, target):
    B = target.shape[0]
    predictions = torch.argmax(prediction, dim=-1).flatten()
    return (predictions == target).sum() / B

@torch.no_grad()
def estimate_loss_bert():
    model.eval()
    running_loss = 0.0
    iters = 0
    correct = 0
    for feature, mask, target in val_dataloader:
        iters += 1
        feature = feature.to(device)
        target = target.to(device)
        mask = mask.to(device)
        with torch.autocast('cuda'):
            output = model.forward(feature, mask)
            loss_val = loss(output, target)
        feature.detach()
        output.detach()
        mask.detach()
        target.detach()
        running_loss += loss_val.item()
        correct += estimate_correct(output, target)
    val_loss = running_loss / iters
    correct /= iters
    model.train()
    return val_loss, correct

def show_progress(epoch, step, total_steps, loss, added_text='', width=30, bar_char='█', empty_char='░'):
    print('\r', end='')
    progress = ""
    for i in range(width):
        progress += bar_char if i < int(step / total_steps * width) else empty_char
    print(f"epoch:{epoch + 1} [{progress}] {step}/{total_steps} loss: {loss:.4f}" + added_text, end='')


def train_model(model, train_dataloader, device, optimizer, scheduler, epochs):
    model.train()
    epoch = 0
    while epoch < epochs:
        running_loss = 0.0
        correct = 0
        start = time.time()
        for i, (feature, mask, target) in enumerate(train_dataloader):
            feature = feature.to(device)
            target = target.to(device)
            mask = mask.to(device)
            with torch.autocast('cuda'):
                output = model.forward(feature, mask)
                loss_val = loss(output, target)
            model.zero_grad()
            loss_val.backward()
            optimizer.step()
            feature.detach()
            target.detach()
            output.detach()
            mask.detach()
            with torch.no_grad():
                running_loss += loss_val.item()
                correct += estimate_correct(output, target)
                if i % 50 == 0:
                    show_progress(epoch, i, len(train_dataloader), running_loss/(i + 1), f" train acc: {correct/(i + 1):.4f}")
            scheduler.step()
        with torch.no_grad():
            correct /= len(train_dataloader)
            val_loss, val_acc = estimate_loss_bert()
            if val_acc > best_val_acc and val_acc > 0.5:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'saved_model.pth') # save model weights
            show_progress(epoch, i, len(train_dataloader), running_loss/(i + 1), f' train acc: {correct:.4f}' + f' val loss: {val_loss:.4f} acc: {val_acc:.4f}' + f' time: {time.time() - start:.4f}')
            print()
        epoch += 1


def evaluate_model(model, test_dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    start = time.time()
    all_preds = []
    all_targets = []

    for i, (feature, mask, target) in enumerate(test_dataloader):
        feature, target, mask = feature.to(device), target.to(device), mask.to(device)
        with torch.no_grad():
            with torch.autocast('cuda'):
                output = model.forward(feature, mask)
                loss_val = loss(output, target)

            running_loss += loss_val.item()
            correct += estimate_correct(output, target)

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % 50 == 0:
                show_progress(0, i, len(test_dataloader), running_loss/(i + 1), f" Test acc: {correct/(i + 1):.4f}")

    correct = np.sum(np.array(all_preds) == np.array(all_targets))
    total = len(all_preds)
    accuracy = correct / total
    show_progress(0, i, len(test_dataloader), running_loss/(i + 1), f' Test acc: {accuracy:.4f}' + f' time: {time.time() - start:.4f}')
    print()
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


def evaluate_sample_predictions(model, test_dataloader, device):
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    model.eval()
    feature, mask, target = next(iter(test_dataloader))
    feature = feature.to(device)
    mask = mask.to(device)
    target = target.to(device)
    output = model(feature, mask)
    output = torch.argmax(output, dim=-1)
    for i in range(10):
        print(tokenizer.decode((feature[i][feature[i]!=0])[1:-1]))
        print()
        print(f'predicted: {emotions[output[i]]} true: {emotions[target[i]]}')
        print()
        print()
    feature.detach()
    mask.detach()
    output.detach()
    target.detach()


if __name__ == "__main__":
    
    train_model(model, train_dataloader, device, optimizer, scheduler, epochs)
