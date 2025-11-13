import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_num = config["general"]["model_num"]
cuda_devices = config["general"]["cuda_devices"]

import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from peft.tuners.lora import Linear as LoRALinear
from torch.utils.data import random_split

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# Pretrained
basemodel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True).to('cuda')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)

# === Unified Dataset ===
class ImageClassificationDataset(Dataset):
    def __init__(self, image_folders, label, processor):
        self.image_paths = []
        self.labels = []
        for folder in image_folders:
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    self.image_paths.append(os.path.join(folder, file))
                    self.labels.append(label)  # ← label duplicated per image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # BCELoss expects float
        return self.image_paths[idx], label


def collate_paths_and_labels(batch):
    # batch is a list of tuples: [(path1, label1), (path2, label2), ...]
    paths = [item[0] for item in batch]  # list of strings
    labels = torch.stack([item[1] for item in batch])  # tensor batch of labels
    return paths, labels

class Classifier(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.processor = processor
        self.base = base_model.to(device)

        # Freeze all DINOv2 parameters
        for param in self.base.parameters():
            param.requires_grad = False
        

        # Explicit classifier layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_paths, return_activations=False):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        # Prepare inputs for X (images) and Y (text)
        X_inputs = self.processor(images=images, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            X_embeddings = self.base.get_image_features(**X_inputs)

        cls_embedding = X_embeddings

        # Layer-by-layer forward
        hidden = self.relu(self.fc1(cls_embedding))
        out = self.sigmoid(self.fc2(hidden)).squeeze()

        if return_activations:
            return out, hidden, cls_embedding  # Return activations if needed
        return out


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for image_paths, labels in val_loader:
        outputs = model(image_paths)
        loss = criterion(outputs, labels.to(device))
        total_loss += loss.item()
        all_preds.append(outputs.cpu())
        all_labels.append(labels.cpu())

    # Optional: compute accuracy
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    predicted_classes = (preds > 0.5).float()
    accuracy = (predicted_classes == labels).float().mean().item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy


# === Training loop
def train(model, loader, epochs=5):
    print(f"Training model for {epochs} epochs")
    model.train()
    max_val_acc = 0
    for epoch in range(epochs):
        total_loss = 0
        for image_paths, labels in loader:
            outputs = model(image_paths)
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(loader)
        val_loss, val_acc = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}")
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            save_model(model)

# === Save & Load
def save_model(model, path=f"clip_classifier.pt"):
    torch.save(model.state_dict(), path)

def load_model(path=f"clip_classifier.pt"):
    model = Classifier(basemodel).to(device)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    wrong_folders = ["your_path_here"]
    corrected_folders = ["your_path_here"]
    augment_folders = ["your_path_here", "your_path_here"]

    wrong_dataset = ImageClassificationDataset(wrong_folders, 1, processor)
    corrected_dataset = ImageClassificationDataset(corrected_folders, 0, processor)
    full_dataset = torch.utils.data.ConcatDataset([wrong_dataset, corrected_dataset])
    # Split into 80% train, 20% val
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    augment_dataset = ImageClassificationDataset(augment_folders, 0, processor)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, augment_dataset])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_paths_and_labels
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_paths_and_labels
    )

    # === Instantiate model
    model = Classifier(basemodel).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train(model, train_loader, epochs=20)
    # save_model(model)
