import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF
sys.path.append(os.path.abspath("misc"))
from accuracies_combiner import process_file

# remove unnessecary stuff
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import struct

torch.set_float32_matmul_precision("high")

# keep at value 5 to have same 5 datasplits in each experiment
number_of_repeats = 5
file_path = "final_tests/mnist/4D/accuracies_rescaled.txt"

# Hyperparameters
HIDDEN_UNITS = 64
BATCH_SIZE = 128
PATIENCE = 15
VAL_SIZE = 10000
MAX_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
def read_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        return data

def read_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

X_temp = read_images("mnist/train-images.idx3-ubyte")
y_temp = read_labels("mnist/train-labels.idx1-ubyte")
X_test = read_images("mnist/t10k-images.idx3-ubyte")
y_test = read_labels("mnist/t10k-labels.idx1-ubyte")

# Rescale features to [-1, +1] linearly per feature
X_temp = (X_temp - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

# PyTorch Dataset
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(1)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_ds = ArrayDataset(X_test, y_test)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

# Model: 2 convolutive layers, 4D-KAF activation
class MNISTCNN(nn.Module):
    def __init__(self, hidden_units, D, grid, no_boundary_nodes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_units, kernel_size=5, padding=2)
        self.act1 = generalized_KAF(hidden_units, 4, D=D, conv=True, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)
        self.conv2 = nn.Conv2d(hidden_units // 4, hidden_units, kernel_size=5, padding=2)
        self.act2 = generalized_KAF(hidden_units, 4, grid_to_copy=self.act1, conv=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(hidden_units // 4 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = self.flatten(x)
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

# for model initialization for each grid type
models = [[7, 'sparse', True, 'sparse_without_boundary_209'], [4, 'uniform', False, 'uniform_256'], [256, 'lattice', False, 'lattice_256'], [256, 'monte_carlo', False, 'monte_carlo_256'], [256, 'p_greedy', False, 'p_greedy_256']]

accuracies_and_epochs = {m[3]: [0, [], [], 0.0, 0.0, 0.0, 0.0] for m in models}

# for number_of_repeats different initializations, compute accuracy for each grid type
for i in range(number_of_repeats):

    # Train / Val / Test split
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=i)

    total_samples = X_train.shape[0]
    train_ds = ArrayDataset(X_train, y_train)
    val_ds = ArrayDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

    for m in models:
        model = MNISTCNN(HIDDEN_UNITS, m[0], m[1], m[2]).to(DEVICE)
        model = torch.compile(model)
        print(model)
        if i == 0:
            accuracies_and_epochs[m[3]][0] = model.act1.dict.shape[0]

        # Loss, optimizer
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

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

        # store accuracy and number of epochs
        accuracies_and_epochs[m[3]][1].append(test_acc)
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