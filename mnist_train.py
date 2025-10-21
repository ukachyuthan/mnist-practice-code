import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ---------------- GPU Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Dataset ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST
full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset       = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split train dataset into train + val + inference-only
total_train = len(full_train_dataset)  # 60,000
val_size = 6000                         # 10% for validation
train_size = total_train - val_size
inference_size = 2000                    # subset for inference
train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])
inference_ds, _ = random_split(train_ds, [inference_size, train_size - inference_size])

# DataLoaders
train_dl      = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_dl        = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
inference_dl  = DataLoader(inference_ds, batch_size=64, shuffle=False, num_workers=2)
test_dl       = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ---------------- CNN Model ----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.fc1   = nn.Linear(128*12*12, 256)  # calculate flattened size
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ---------------- Optimizer and Loss ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------------- Training Loop ----------------
epochs = 10
best_val_loss = float("inf")
os.makedirs("models", exist_ok=True)

for epoch in range(epochs):
    # ---- Training ----
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1} Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_dl)
    print(f"Epoch {epoch+1} avg training loss: {avg_train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_dl)
    val_acc = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    # ---- Save best model ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pt")
        print("Saved new best model!")

# ---------------- Inference on holdout dataset ----------------
model.load_state_dict(torch.load("models/best_model.pt"))
model.eval()
all_preds = []
with torch.no_grad():
    for imgs, labels in inference_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
print(f"Inference done on holdout set ({len(all_preds)} samples)")

# ---------------- Optional: Visualize some predictions ----------------
imgs, labels = next(iter(inference_dl))
imgs, labels = imgs.to(device), labels.to(device)
outputs = model(imgs)
_, preds = torch.max(outputs, 1)

fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i].cpu().squeeze(), cmap='gray')
    ax.set_title(f"Pred: {preds[i].item()}, Label: {labels[i].item()}")
    ax.axis('off')
plt.show()
