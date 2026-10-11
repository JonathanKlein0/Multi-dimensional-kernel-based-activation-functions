import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF
sys.path.append(os.path.abspath("misc"))
from accuracies_combiner import process_file

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
INPUT_DIM = 48
HIDDEN_UNITS = 100
BATCH_SIZE = 100
L2_FACTOR = 1e-4
PATIENCE = 15
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MAX_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
def load_sensorless():
    csv_path = "sensorless/sensorless.csv"
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X.values.astype(np.float32), y.values

X, y = load_sensorless()

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

# Rescale features to [-1, +1] linearly per feature
X_min = X.min(axis=0)
X_max = X.max(axis=0)
scale = (X_max - X_min)
X_scaled = 2.0 * (X - X_min) / scale - 1.0
X = X_scaled.astype(np.float32)

# PyTorch Dataset
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model: one hidden layer, 2D-KAF
class SensorlessNet(nn.Module):
    def __init__(self, input_dim, hidden_units, num_classes, D, grid, no_boundary_nodes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.act = generalized_KAF(hidden_units, 2, D=D, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)
        self.fc2 = nn.Linear(hidden_units // 2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        logits = self.fc2(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# for model initialization for each grid type
models = [[10, 'uniform', False, 'uniform_0.001'], [10, 'uniform', False, 'uniform_1000']]
c = 1/(0.6**2) # 0.6 = grid distance along the axis
gamma_list = [c / i for i in [1/1000, 1000]]

accuracies_and_epochs = {m[3]: [0, [], [], 0.0, 0.0, 0.0, 0.0] for m in models}

# for 5 different data splits, compute accuracies for each grid type
for i in range(5):

    # Train / Val / Test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_enc, test_size=TEST_FRACTION, stratify=y_enc)
    val_relative = VAL_FRACTION / (1.0 - TEST_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative, stratify=y_temp)

    train_ds = ArrayDataset(X_train, y_train)
    val_ds = ArrayDataset(X_val, y_val)
    test_ds = ArrayDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    j = 0

    for m in models:
        model = SensorlessNet(INPUT_DIM, HIDDEN_UNITS, num_classes, m[0], m[1], m[2]).to(DEVICE)
        model.act.gamma = gamma_list[j]
        j += 1
        print(model)
        if i == 0:
            accuracies_and_epochs[m[3]][0] = model.act.dict.shape[0]

        # Loss, optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=L2_FACTOR)

        # Training loop with early stopping based on validation accuracy
        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0
        epochs_till_best = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            batches = 0
            epochs_till_best += 1
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batches += 1

            avg_train_loss = running_loss / max(1, batches)

            # Validation accuracy
            val_acc = evaluate_accuracy(model, val_loader, DEVICE)
            print(f"Epoch {epoch:03d} | Train loss: {avg_train_loss:.4f} | Val acc: {val_acc*100:.2f}%")

            # Early stopping decision
            if val_acc > best_val_acc + 1e-12:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping: no improvement in validation accuracy for {PATIENCE} consecutive epochs.")
                epochs_till_best -= PATIENCE
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Restored model with best validation accuracy = {best_val_acc*100:.2f}%")

        # Final test accuracy
        test_acc = evaluate_accuracy(model, test_loader, DEVICE)
        print(f"Test accuracy: {test_acc*100:.2f}% after {epochs_till_best} epochs")

        # store accuracy and number of epochs
        accuracies_and_epochs[m[3]][1].append(test_acc)
        accuracies_and_epochs[m[3]][2].append(epochs_till_best)

# compute mean and std for accuracies
for m in models:
    accuracies_and_epochs[m[3]][3] = np.mean(accuracies_and_epochs[m[3]][1])
    accuracies_and_epochs[m[3]][4] = np.std(accuracies_and_epochs[m[3]][1])
    accuracies_and_epochs[m[3]][5] = np.mean(accuracies_and_epochs[m[3]][2])
    accuracies_and_epochs[m[3]][6] = np.std(accuracies_and_epochs[m[3]][2])

# write data into file
with open("other_stuff/bandwidth_choice_3.txt", "r") as f:
    empty = (f.read(1) == "")

with open("other_stuff/bandwidth_choice_3.txt", "a") as f:
    if empty:
        f.write("Grid type: number of grid points / accuracies in each run / epochs in each run / mean accuracy / std of accuracy / mean epochs / std of epochs\n")
        for key, value in accuracies_and_epochs.items():
            f.write(f"{key}: {value}\n")
    else:
        for key, value in accuracies_and_epochs.items():
            f.write(f"{key}: {value}\n")

if not empty:
    process_file("other_stuff/bandwidth_choice_2.txt")