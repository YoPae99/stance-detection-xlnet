import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
MAX_LEN = 128
BATCH_SIZE = 16
CHECKPOINT_PATH = 'checkpoint.pt'

# Label mapping
label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
inv_label_map = {v: k for k, v in label_map.items()}

# Load tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=4)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# Load and preprocess test data
test_stances_path = '/content/drive/MyDrive/FakeNewsDetection/data/competition_test_stances.csv'
test_bodies_path = '/content/drive/MyDrive/FakeNewsDetection/data/competition_test_bodies.csv'

stances_df = pd.read_csv(test_stances_path)
bodies_df = pd.read_csv(test_bodies_path)
df = pd.merge(stances_df, bodies_df, on='Body ID')
df['text'] = df['Headline'] + " " + df['articleBody']
df['label'] = df['Stance'].map(label_map)
df = df.dropna(subset=['text', 'label'])

texts = df['text'].tolist()
labels = df['label'].tolist()

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

test_dataset = NewsDataset(texts, labels, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Evaluation
all_preds, all_targets = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_batch = batch['labels'].detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(labels_batch)

# Accuracy
accuracy = accuracy_score(all_targets, all_preds)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Class-wise Accuracy
class_accuracies = {}
for label, label_id in label_map.items():
    class_correct = sum(1 for pred, true in zip(all_preds, all_targets) if pred == true == label_id)
    class_total = sum(1 for true in all_targets if true == label_id)
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    class_accuracies[label] = class_accuracy

print("\nClass-wise Accuracy:")
for label, accuracy in class_accuracies.items():
    print(f"{label}: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=label_map.keys()))
