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

# keep at value 5 to have same 5 datasplits in each experiment
number_of_repeats = 5
file_path = "final_tests/misc/optimizer_accuracies_conv.txt"

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
class CIFAR10CNN(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_units, kernel_size=5, padding=2)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(hidden_units, hidden_units, kernel_size=5, padding=2)
        self.act2 = nn.ELU()
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

# different optimizers
optimizers = []

algorithms = ["Adam", "Adamw"]
lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

for alg in algorithms:
    for lr in lrs:
        for weight_decay in weight_decays:
            optimizers.append((alg, lr, weight_decay))

accuracies_and_epochs = {opt: [0, [], [], 0.0, 0.0, 0.0, 0.0] for opt in optimizers}

# for number_of_repeats different initializations, compute accuracy for each grid type
for i in range(number_of_repeats):

    # Train / Val / Test split
    total_samples = TRAIN_SIZE
    train_ds, val_ds = random_split(full_train_dataset, [TRAIN_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(i))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

    for opt in optimizers:
        model = CIFAR10CNN(HIDDEN_UNITS).to(DEVICE)
        print(model)

        # Loss, optimizer
        criterion = nn.CrossEntropyLoss(reduction='sum')
        if opt[0] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=opt[1], weight_decay=opt[2])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=opt[1], weight_decay=opt[2])

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
        accuracies_and_epochs[opt][1].append(test_acc)
        accuracies_and_epochs[opt][2].append(epochs_till_best)

        # free GPU cache
        del model
        torch.cuda.empty_cache()

# compute mean and std for accuracies
for opt in optimizers:
    accuracies_and_epochs[opt][3] = float(np.mean(accuracies_and_epochs[opt][1]))
    accuracies_and_epochs[opt][4] = float(np.std(accuracies_and_epochs[opt][1]))
    accuracies_and_epochs[opt][5] = float(np.mean(accuracies_and_epochs[opt][2]))
    accuracies_and_epochs[opt][6] = float(np.std(accuracies_and_epochs[opt][2]))

# write data into file
with open(file_path, "r") as f:
    empty = (f.read(1) == "")

with open(file_path, "a") as f:
    if empty:
        f.write("Optimizer, lr, weight_decay: accuracies in each run / epochs in each run / mean accuracy / std of accuracy / mean epochs / std of epochs\n")
        for key, value in accuracies_and_epochs.items():
            f.write(f"{key}: {value}\n")
    else:
        for key, value in accuracies_and_epochs.items():
            f.write(f"{key}: {value}\n")

if not empty:
    process_file(file_path)