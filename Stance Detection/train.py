import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetForSequenceClassification, get_scheduler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
PATIENCE = 2

# Load tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Load and merge FNC-1 dataset
base_dir = os.path.join(os.path.dirname(__file__), 'Data')
stances_path = os.path.join(base_dir, 'train_stances.csv')
bodies_path = os.path.join(base_dir, 'train_bodies.csv')


stances_df = pd.read_csv(stances_path)
bodies_df = pd.read_csv(bodies_path)

df = pd.merge(stances_df, bodies_df, on='Body ID')
df['text'] = df['Headline'] + " " + df['articleBody']

# Encode labels
label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
df['label'] = df['Stance'].map(label_map)
df = df.dropna(subset=['text', 'label'])

# Split
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load model with 4 labels for stance detection
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=4)
model.to(device)

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Loss
criterion = nn.CrossEntropyLoss()

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Initialize early stopping
early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training
    model.train()
    train_loss = 0
    train_preds, train_targets = [], []

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        train_preds.extend(preds)
        train_targets.extend(labels)

    train_acc = accuracy_score(train_targets, train_preds)
    print(f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc * 100:.2f}%")

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = batch['labels'].detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels)

    val_acc = accuracy_score(val_targets, val_preds)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    # Early stopping
    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

# Load the best model
model.load_state_dict(torch.load('checkpoint.pt'))
print("Training complete. Best model loaded.")