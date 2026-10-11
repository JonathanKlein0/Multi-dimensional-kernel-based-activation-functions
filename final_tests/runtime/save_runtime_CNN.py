import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF
sys.path.append(os.path.abspath("misc"))
from accuracies_combiner import process_file

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import struct
from torchvision import datasets, transforms

import time

torch.set_float32_matmul_precision("high")

# Hyperparameters
HIDDEN_UNITS = 152
BATCH_SIZE = 128
PATIENCE = 15
TRAIN_SIZE = 40000
VAL_SIZE = 10000
MAX_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

full_train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

# Model: 2 convolutive layers, 2D-KAF activation
class CIFAR10CNN2D(nn.Module):
    def __init__(self, hidden_units, D, grid, no_boundary_nodes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_units, kernel_size=5, padding=2)
        self.act1 = generalized_KAF(hidden_units, 2, D=D, conv=True, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)
        self.conv2 = nn.Conv2d(hidden_units // 2, hidden_units, kernel_size=5, padding=2)
        self.act2 = generalized_KAF(hidden_units, 2, grid_to_copy=self.act1, conv=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(hidden_units // 2 * 7 * 7, 10)

    def forward(self, x):
        x = self.dropout(self.pool(self.act1(self.conv1(x))))
        x = self.dropout(self.pool(self.act2(self.conv2(x))))
        x = self.flatten(x)
        logits = self.output(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Model: 2 convolutive layers, 4D-KAF activation
class CIFAR10CNN4D(nn.Module):
    def __init__(self, hidden_units, D, grid, no_boundary_nodes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_units, kernel_size=5, padding=2)
        self.act1 = generalized_KAF(hidden_units, 4, D=D, conv=True, boundary=3.0, grid=grid, no_boundary_nodes=no_boundary_nodes)
        self.conv2 = nn.Conv2d(hidden_units // 4, hidden_units, kernel_size=5, padding=2)
        self.act2 = generalized_KAF(hidden_units, 4, grid_to_copy=self.act1, conv=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(hidden_units // 4 * 7 * 7, 10)

    def forward(self, x):
        x = self.dropout(self.pool(self.act1(self.conv1(x))))
        x = self.dropout(self.pool(self.act2(self.conv2(x))))
        x = self.flatten(x)
        logits = self.output(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Model: 2 convolutive layers, ReLU activation
class CIFAR10CNNReLU(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_units, kernel_size=5, padding=2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_units, hidden_units, kernel_size=5, padding=2)
        self.act2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(hidden_units * 7 * 7, 10)

    def forward(self, x):
        x = self.dropout(self.pool(self.act1(self.conv1(x))))
        x = self.dropout(self.pool(self.act2(self.conv2(x))))
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

# Train / Val / Test split
total_samples = TRAIN_SIZE
train_ds, val_ds = random_split(full_train_dataset, [TRAIN_SIZE, VAL_SIZE])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

dict_sizes = [200]# [100, 120, 140, 160, 180, 200]
times_2D, times_4D = [], []
model_sizes_2D, model_sizes_4D = [], []

for i in [1]:
    for D in dict_sizes:
        if i == 0:
            model = CIFAR10CNN2D(HIDDEN_UNITS, D, "monte_carlo", False).to(DEVICE)
            temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_sizes_2D.append(temp)
        else:
            model = CIFAR10CNN4D(HIDDEN_UNITS, D, "monte_carlo", False).to(DEVICE)
            temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_sizes_4D.append(temp)
        model = torch.compile(model)
        
        # Loss, optimizer
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

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
            val_acc = evaluate_accuracy(model, val_loader, DEVICE)
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
            
        if i == 0:
            print(f"2D-M-KAF of size {D}: {total / 3} with {temp} parameters")
        else:
            print(f"4D-M-KAF of size {D}: {total / 3} with {temp} parameters")
        # free GPU cache
        del model
        torch.cuda.empty_cache()

np.savez("final_tests/runtime/runtime_CNN.npz", dict_sizes=dict_sizes, times_2D=times_2D, times_4D=times_4D, model_sizes_2D=model_sizes_2D, model_sizes_4D=model_sizes_4D)