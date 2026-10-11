import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF

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
HIDDEN_UNITS = 128
BATCH_SIZE = 128
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

# Model: 1 hidden layer, 2D-KAF activation
class SensorlessNet(nn.Module):
    def __init__(self, input_dim, hidden_units, D, grid, no_boundary_nodes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.output = nn.Linear(hidden_units // 2, num_classes)
        self.act = generalized_KAF(hidden_units, 2, D=D, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)

    def forward(self, x):
        x = self.act(self.fc1(x))
        logits = self.output(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Helper: Evaluate accuracy
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return correct / total

X_temp, X_test, y_temp, y_test = train_test_split(X, y_enc, test_size=TEST_FRACTION, stratify=y_enc)
val_relative = VAL_FRACTION / (1.0 - TEST_FRACTION)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative, stratify=y_temp)

total_samples = X_train.shape[0]

train_ds = ArrayDataset(X_train, y_train)
val_ds = ArrayDataset(X_val, y_val)
test_ds = ArrayDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

model = SensorlessNet(INPUT_DIM, HIDDEN_UNITS, 10, 'uniform', False).to(DEVICE)
old_alpha = model.act.alpha.data.clone()
# Loss, optimizer
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)

pre_acts_init = []

def save_pre_activation(module, inputs, output):
    pre_acts_init.append(output.detach().cpu())

handle = model.fc1.register_forward_hook(save_pre_activation)

model.eval()
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(DEVICE)
        model(x)

handle.remove()

# Combine batches
pre_acts_init = torch.cat(pre_acts_init, dim=0)

# Training loop with early stopping based on validation area under curve
best_val_acc = 0.0
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
    val_acc = evaluate_accuracy(model, val_loader, DEVICE)
    print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val acc: {val_acc*100:.2f}%")

    # Early stopping decision
    if val_acc > best_val_acc + 1e-12:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
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

pre_acts_trained = []

def save_pre_activation_trained(module, inputs, output):
    pre_acts_trained.append(output.detach().cpu())

handle = model.fc1.register_forward_hook(save_pre_activation_trained)

model.eval()
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(DEVICE)
        model(x)

handle.remove()

# Combine batches
pre_acts_trained = torch.cat(pre_acts_trained, dim=0)

torch.save({'pre_acts_trained': pre_acts_trained, 'pre_acts_init': pre_acts_init, 'old_alpha': old_alpha, 'new_alpha': model.act.alpha.data, 'gamma': model.act.gamma, 'dict': model.act.dict.data}, 'before_and_after_training.pt')