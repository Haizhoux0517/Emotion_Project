from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# HYPERPARAMETERS
HP = {
    "BATCH_SIZE": 16,           # TODO: Fine-tune [16, 32, 64, 128] Less than 10000: try 16 - 32; More than 10000: try 64 - 128.
    "LEARNING_RATE": 2e-5,      # TODO: Fine-tune [1e-5, 2e-5, 0.001]
    "NUM_EPOCHS": 5,            # TODO: Fine-tune [1, 5, 8, 10, 20]
    "HIDDEN_SIZE": 256,         # TODO: Fine-tune [64, 128, 256, 512] Small dataset: try 64 - 256; Large dataset: try 256.
    "NUM_LAYERS": 4,            # TODO: Fine-tune [1, 2, 3, 4]. Number of layers in the Bi-GRU layer.
    "MAX_LENGTH": 128,          # Most of the reviews should be less than 128 tokens (DistilBERT can tokenize up to 512).
}


def data_balancing(csv_file="full.csv", ratio=1.0):
    '''
    Resample data to ensure balanced data source (if using full version of dataset, the number of entries after resampling is around 90000).

    Parameters:
    - csv_file: String. CSV file that contains data source.
                Defaults to "full.csv" (entire emotion dataset).
    - ratio: Float (optional). The ratio of data to select from the source dataset.
             Should be a value between 0.0 and 1.0, where 1.0 selects all data.
             Defaults to 1.0 (select all data).

    Returns:
    - data_balanced: Pandas DataFrame, the balanced data frame after resampling.
    '''
    # Read data from CSV file.
    data = pd.read_csv(csv_file)
    # Separate data by class
    classes = data['label'].unique()
    data_by_class = {emotion: data[data['label'] == emotion] for emotion in classes}
    # Find the number of samples in the minority class.
    minority_class = min(len(data_by_class[emotion]) for emotion in classes)
    # Decide the number of data to keep for each class.
    number_per_class = int(minority_class * ratio)
    # Resample each class to match the number of samples in the minority class.
    data_resampled = {}
    for emotion in classes:
        data_resampled[emotion] = resample(data_by_class[emotion], replace=True, n_samples=number_per_class, random_state=42)
    # Combine resampled dataframes into a single dataframe.
    data_balanced = pd.concat(list(data_resampled.values()))
    # Shuffle the dataframe to ensure randomness
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return data_balanced


def data_preprocessing(data):
    '''
    Perform data preprocessing.

    Parameters:
    - data: Pandas DataFrame. Balanced data frame after resampling.

    Returns:
    - data: Pandas DataFrame. Sanitized data after preprocessing.
    '''
    # Dropping the index colums.
    data.drop('Unnamed: 0', inplace=True, axis=1)
    # Preprocess: 1. Remove urls.
    data['text'] = data['text'].str.replace(r'http\S+', '', regex=True)
    # Preprocess: 2. Remove special characters and punctuation: Negates any character that is not in the set of word characters and whitespace characters.
    data['text'] = data['text'].str.replace(r'[^\w\s]', '', regex=True)
    # Preprocess: 3. Remove extra whitespaces.
    data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True)
    # Preprocess: 4. Remove numeric values.
    data['text'] = data['text'].str.replace(r'\d+', '', regex=True)
    # Preprocess: 5. Lowercasing
    data['text'] = data['text'].str.lower()
    # Preprocess: 6. Remove stop words in English: Reduce noise word like "the", "is", "and", etc. 
    stop = set(stopwords.words('english'))
    data["text"] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    # Preprocess: 7. Remove non-alphanumeric characters.
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    return data


# Dataset
class EmotionDataset(Dataset):
    '''
    Dataset class defines how the text-label is pre-processed before sending it to the model.
    '''
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'text']
        label = self.data.loc[idx, 'label']
        encoded_inputs = self.tokenizer(text, 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=self.max_length, 
                                        return_tensors='pt')
        # Note: Can return token_type_ids by setting return_token_type_ids to True.
        # For our task, token_type_ids is optional because our classification task only requires one sequence as input.
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        
        return {
            'input_ids': input_ids,           
            'attention_mask': attention_mask,
            'label': label
        }

# Model architecture
class DistilBERTBiGRU(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_layers, num_classes):
        super(DistilBERTBiGRU, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)  
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 since it is bidirectional.
        self.activation = nn.Tanh()     # TODO: Fine-tuning: [Sigmoid, Tanh, softmax]
        # self.activation = nn.Sigmoid()
        # self.activation = nn.functional.softmax     

    def forward(self, input_ids, attention_mask):
        # DistilBERT Layer
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]  # Last hidden state: (batch_size, sequence_length, hidden_size)
        
        # Bi-GRU Layer
        gru_output, _ = self.gru(bert_output)       # Output: (batch_size, seq_length, num_direction * hidden_size)
 
        # Option 1: Selects the last hidden state for each sequence in the batch. 
        # The slicing operation [:, -1, :] selects all batches (:), the last time step (-1), and all elements along the last hidden state (:).
        last_hidden_output = gru_output[:, -1, :]   # Output: (batch_size, num_direction * hidden_size)
        
        # Option 2: Element-wise Max Pooling
        max_pooling_output, _ = torch.max(gru_output, dim=1)  # Max pooling along the sequence dimension

        # Option 3: Mean Pooling
        mean_pooling_output = torch.mean(gru_output, dim=1)  # Mean pooling along the sequence dimension
        
        # Dropout Layer
        dropout_output = self.dropout(mean_pooling_output)   # TODO: Fine-tuning: [last_hidden_output, max_pooling_output, mean_pooling_output]

        # Classification Layer
        output = self.classifier(dropout_output)            # Output: (batch_size, num_classes)
 
        # Activation Layer  
        probabilities = self.activation(output)     # Output: (batch_size, num_classes)

        return probabilities


# Train model
def train(model, train_loader, optimizer, criterion, device, epoch):
    epoch_loss = 0
    label_trues = []
    label_preds = []

    # Set the model to training mode.
    model.train()
    
    # Iterate over the training dataset
    for _, data in tqdm(enumerate(train_loader, 0), desc="Training"):
        # Move batch to device.
        input_ids = data['input_ids'].squeeze(1).to(device)
        attention_mask = data['attention_mask'].squeeze(1).to(device)
        label = data['label'].to(device) 
        
        # Forward pass.
        outputs = model(input_ids, attention_mask)
        
        # Zero the gradients.
        optimizer.zero_grad()
        
        # Compute the loss.
        loss = criterion(outputs, label)
        epoch_loss += loss.item()
        
        if _%50 == 0:
            print(f'Epoch: {epoch+1}, Loss:  {loss.item()}')
        
        # Backward pass.
        loss.backward()
        
        # Update the model parameters.
        optimizer.step()

        # Find the indices of the maximum values along dim=1 (across classes).
        preds = torch.argmax(outputs, axis=1)
            
        label_preds.extend(preds.cpu().detach().numpy().tolist())
        label_trues.extend(label.cpu().detach().numpy().tolist())
        
    # Calcualte test loss.
    loss = epoch_loss / len(train_loader)

    # Evaluate on accuracy, precision, recall and F1 score.
    accuracy = metrics.accuracy_score(label_trues, label_preds)

    return loss, accuracy


# Evaluate model
def evaluate(model, test_loader, criterion, device):
    epoch_loss = 0
    label_trues = []
    label_preds = []

    # Set the model to evaluation mode.
    model.eval()
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0), desc="Evaluating"):
            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].squeeze(1).to(device)
            label = data['label'].to(device)
            
            # Predicted output.
            outputs = model(input_ids, attention_mask)

            # Compute the loss.
            loss = criterion(outputs, label)
            epoch_loss += loss.item()
            
            # Find the indices of the maximum values along dim=1 (across classes).
            preds = torch.argmax(outputs, axis=1)
            
            label_preds.extend(preds.cpu().detach().numpy().tolist())
            label_trues.extend(label.cpu().detach().numpy().tolist())
    
    # Calcualte test loss.
    loss = epoch_loss / len(test_loader)

    # Evaluate on accuracy, precision, recall and F1 score.
    accuracy = metrics.accuracy_score(label_trues, label_preds)
    precision = metrics.precision_score(label_trues, label_preds, average="macro", zero_division=0.0)
    recall = metrics.recall_score(label_trues, label_preds, average="macro", zero_division=0.0)
    f1 = metrics.f1_score(label_trues, label_preds, average="macro", zero_division=0.0)
    
    return loss, accuracy, precision, recall, f1


# Main code
if __name__ == "__main__":
    # Step 1: Setting up the device for GPU usage.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 2: Load and preprocess the dataset.
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    raw_data = data_balancing("full.csv", 1)    # 1/8: 11226
    data = data_preprocessing(raw_data)
    print("# of Data Entries: "+ str(len(data)))
     
    # Step 3: Split train and test set.
    dataset = EmotionDataset(data, tokenizer, max_length=HP["MAX_LENGTH"])
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Step 4: Data loader: Feed the data in batches to the network for suitable training and processing.
    train_loader = DataLoader(train_data, batch_size=HP["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=HP["BATCH_SIZE"], shuffle=True)
    print("Train Dataset: {}".format(len(train_data)))
    print("Test Dataset: {}".format(len(test_data)))
    
    # Step 5: Define BERT-BiGRU model architecture.
    model = DistilBERTBiGRU(bert_model_name='distilbert-base-uncased', hidden_size=HP["HIDDEN_SIZE"], num_layers=HP["NUM_LAYERS"], num_classes=6)
    model.to(device)
    
    # Step 6: Train the model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["LEARNING_RATE"])   # TODO: Fine-tuning: [Adam, AdamW, SGD] 
    criterion = nn.CrossEntropyLoss()
    
    epochs = HP["NUM_EPOCHS"]
    history = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, test_loader, criterion, device)

        # Create a dictionary for the current epoch
        history_entry = {}

        # Update the dictionary with the metrics
        history_entry['train_loss'] = train_loss
        history_entry['test_loss'] = val_loss
        history_entry['train_acc'] = train_acc
        history_entry['test_acc'] = val_acc

        # Append the dictionary to the history list
        history.append(history_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "distilbert-bigru-emotion.pt")

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'\t Average Train Loss: {train_loss:.4f} |  Train Accuracy: {train_acc:.4f}')
        print(f'\t Average Val Loss: {val_loss:.4f} |  Val Accuracy: {val_acc:.4f}')
        print(f'\t Val F1 Score: {val_f1:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}')

        # Acuuracy benchmark with hugging face: 0.927
    
