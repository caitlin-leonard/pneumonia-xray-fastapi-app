import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ----- 1. Dataset paths -----
DATASET_ROOT = "/Users/caitlinleonard/.cache/kagglehub/datasets/divyam6969/chest-xray-pneumonia-dataset/versions/1"
train_dir = os.path.join(DATASET_ROOT, "train")

# ----- 2. Transforms & dataset -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # X-rays are 1-channel; ResNet expects 3
    transforms.ToTensor(),
])

full_ds = datasets.ImageFolder(train_dir, transform=transform)
num_classes = len(full_ds.classes)
print("Classes:", full_ds.classes)
print("Total images:", len(full_ds))

# simple 80/20 train/val split
val_size = int(0.2 * len(full_ds))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# ----- 3. Model (ResNet18) -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# replace final layer for our 3 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# ----- 4. Loss & optimizer -----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----- 5. Training loop -----
EPOCHS = 3  # start small

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
    )

# ----- 6. Save model -----
os.makedirs("models", exist_ok=True)
save_path = os.path.join("models", "pneumonia_resnet18.pth")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": full_ds.classes,
    },
    save_path,
)
print("Saved model to:", save_path)
