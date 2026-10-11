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

torch.set_float32_matmul_precision("high")

number_of_repeats = 5
file_path = "final_tests/susy/4D/accuracies_1.txt"

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
    
# Model: 1 hidden layer, 4D-KAF activation
class SUSYNet(nn.Module):
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

# for model initialization for each grid type
models = [[7, 'sparse', True, 'sparse_without_boundary_209'], [4, 'uniform', False, 'uniform_256'], [256, 'lattice', False, 'lattice_256'], [256, 'monte_carlo', False, 'monte_carlo_256'], [256, 'p_greedy', False, 'p_greedy_256']]

accuracies_and_epochs = {m[3]: [0, [], [], 0.0, 0.0, 0.0, 0.0] for m in models}

# for number_of_repeats different initializations, compute AUC for each grid type
for i in range(number_of_repeats):

    for m in models:
        model = SUSYNet(INPUT_DIM, HIDDEN_UNITS, m[0], m[1], m[2]).to(DEVICE)
        model = torch.compile(model)
        print(model)
        if i == 0:
            accuracies_and_epochs[m[3]][0] = model.act.dict.shape[0]

        # Loss, optimizer
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)

        # Training loop with early stopping based on validation area under curve
        best_val_auc = 0.0
        best_state = None
        epochs_no_improve = 0
        epochs_till_best = 0

        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(1, MAX_EPOCHS + 1):
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
            print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}")

            # Early stopping decision
            if val_auc > best_val_auc + 1e-12:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping: no improvement in validation AUC for {PATIENCE} consecutive epochs.")
                epochs_till_best -= PATIENCE
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Restored model with best validation area under the curve = {best_val_auc*100:.2f}%")

        # Final test accuracy
        test_acc, test_auc = evaluate_accuracy_auc(model, test_loader, DEVICE)
        print(f"Test accuracy: {test_acc*100:.2f}% | Test AUC: {test_auc:.4f} after {epochs_till_best} epochs")

        # store accuracy and number of epochs
        accuracies_and_epochs[m[3]][1].append(test_auc)
        accuracies_and_epochs[m[3]][2].append(epochs_till_best)

        # free GPU cache
        del model
        torch.cuda.empty_cache()

    # compute mean and std for accuracies
    for m in models:
        accuracies_and_epochs[m[3]][3] = float(np.mean(accuracies_and_epochs[m[3]][1]))
        accuracies_and_epochs[m[3]][4] = float(np.std(accuracies_and_epochs[m[3]][1]))
        accuracies_and_epochs[m[3]][5] = float(np.mean(accuracies_and_epochs[m[3]][2]))
        accuracies_and_epochs[m[3]][6] = float(np.std(accuracies_and_epochs[m[3]][2]))

    # write data into file
    with open(file_path, "r") as f:
        empty = (f.read(1) == "")

    with open(file_path, "a") as f:
        if empty:
            f.write("Grid type: number of grid points / accuracies in each run / epochs in each run / mean accuracy / std of accuracy / mean epochs / std of epochs\n")
            for key, value in accuracies_and_epochs.items():
                f.write(f"{key}: {value}\n")
        else:
            for key, value in accuracies_and_epochs.items():
                f.write(f"{key}: {value}\n")

    if not empty:
        process_file(file_path)

    accuracies_and_epochs = {m[3]: [0, [], [], 0.0, 0.0, 0.0, 0.0] for m in models}