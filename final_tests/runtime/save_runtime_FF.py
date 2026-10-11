import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF
sys.path.append(os.path.abspath("misc"))
from accuracies_combiner import process_file

from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time

torch.set_float32_matmul_precision("high")

# Hyperparameters
INPUT_DIM = 18
HIDDEN_UNITS = 320
BATCH_SIZE = 1024
PATIENCE = 15
MAX_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SUSY data
def load_susy():
    csv_path = "susy/SUSY.csv.gz"
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values.astype(np.int64)
    return X, y

X, y = load_susy()

# Feature scaling to [-1, +1]
X_min = X.min(axis=0)
X_max = X.max(axis=0)
scale = (X_max - X_min)
X_scaled = 2.0 * (X - X_min) / scale - 1.0
X = X_scaled.astype(np.float32)

# PyTorch Dataset
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Train / Val / Test split
X_train, X_val, X_test = X[0:4_000_000], X[4_000_000:4_500_000], X[4_500_000:5_000_000]
y_train, y_val, y_test = y[0:4_000_000], y[4_000_000:4_500_000], y[4_500_000:5_000_000]

total_samples = X_train.shape[0]

X_train, X_val, X_test = torch.from_numpy(X_train).float(), torch.from_numpy(X_val).float(), torch.from_numpy(X_test).float()
y_train, y_val, y_test = torch.from_numpy(y_train).long(), torch.from_numpy(y_val).long(), torch.from_numpy(y_test).long()

train_ds = ArrayDataset(X_train, y_train)
val_ds = ArrayDataset(X_val, y_val)
test_ds = ArrayDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    
# Model: 1 hidden layer, 2D-KAF activation
class SUSYNet2D(nn.Module):
    def __init__(self, input_dim, hidden_units, D, grid, no_boundary_nodes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.output = nn.Linear(hidden_units // 2, 2)
        self.act = generalized_KAF(hidden_units, 2, D=D, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)

    def forward(self, x):
        x = self.act(self.fc1(x))
        logits = self.output(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Model: 1 hidden layer, 4D-KAF activation
class SUSYNet4D(nn.Module):
    def __init__(self, input_dim, hidden_units, D, grid, no_boundary_nodes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.output = nn.Linear(hidden_units // 4, 2)
        self.act = generalized_KAF(hidden_units, 4, D=D, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)

    def forward(self, x):
        x = self.act(self.fc1(x))
        logits = self.output(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Model: 1 hidden layer, ReLU activation
class SUSYNetReLU(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.output = nn.Linear(hidden_units, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        logits = self.output(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Helper: Evaluate accuracy
def evaluate_accuracy_auc(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_probs = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.append(yb.cpu().numpy())
            y_probs.append(probs.cpu().numpy())

    acc = correct / total
    y_true = np.concatenate(y_true)
    y_probs = np.concatenate(y_probs)

    auc = roc_auc_score(y_true, y_probs)
    return acc, auc

dict_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
times_2D, times_4D = [], []
model_sizes_2D, model_sizes_4D = [], []

progress = 5

model = SUSYNetReLU(INPUT_DIM, HIDDEN_UNITS).to(DEVICE)
model_size_relu = [sum(p.numel() for p in model.parameters() if p.requires_grad)]
model = torch.compile(model)
        
# Loss, optimizer
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)

# Training loop with early stopping based on validation area under curve
best_val_auc = 0.0
best_state = None
epochs_no_improve = 0
epochs_till_best = 0

scaler = torch.amp.GradScaler('cuda')

for epoch in range(5):
    torch.cuda.synchronize()
    start = time.time()
    model.train()
    running_loss = 0.0
    epochs_till_best += 1
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.detach()

    train_loss = running_loss.item() / total_samples

    # Validation accuracy
    val_acc, val_auc = evaluate_accuracy_auc(model, val_loader, DEVICE)
    torch.cuda.synchronize()
    stop = time.time()
    if epoch == 1:
        total = (stop - stop)
    if epoch > 1:
        total += (stop - start)

time_relu = [total / 3]
        
print(f"ReLU: {total / 3}")

for i in range(2):
    for D in dict_sizes:
        if i == 0:
            model = SUSYNet2D(INPUT_DIM, HIDDEN_UNITS, D, "monte_carlo", False).to(DEVICE)
            model_sizes_2D.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        else:
            model = SUSYNet4D(INPUT_DIM, HIDDEN_UNITS, D, "monte_carlo", False).to(DEVICE)
            model_sizes_4D.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.compile(model)
        
        # Loss, optimizer
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)

        # Training loop with early stopping based on validation area under curve
        best_val_auc = 0.0
        best_state = None
        epochs_no_improve = 0
        epochs_till_best = 0

        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(5):
            torch.cuda.synchronize()
            start = time.time()
            model.train()
            running_loss = 0.0
            epochs_till_best += 1
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.detach()

            train_loss = running_loss.item() / total_samples

            # Validation accuracy
            val_acc, val_auc = evaluate_accuracy_auc(model, val_loader, DEVICE)
            torch.cuda.synchronize()
            stop = time.time()
            if epoch == 1:
                total = (stop - stop)
            if epoch > 1:
                total += (stop - start)
        if i == 0:
            times_2D.append(total / 3)
        else:
            times_4D.append(total / 3)
            
        print(f"Progress = {progress}%")
        if i == 0:
            print(f"2D-M-KAF of size {D}: {total / 3}")
        else:
            print(f"4D-M-KAF of size {D}: {total / 3}")
        progress += 5

np.savez("final_tests/runtime/runtime_FF.npz", dict_sizes=dict_sizes, times_2D=times_2D, times_4D=times_4D, model_sizes_2D=model_sizes_2D, model_sizes_4D=model_sizes_4D, time_relu=time_relu, model_size_relu= model_size_relu)