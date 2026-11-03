import json

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def code_cell(source):
    lines = source.strip().split("\n")
    cleaned = []
    for l in lines:
        stripped = l.strip()
        if stripped.startswith("# Question") or stripped.startswith("# Case") or stripped.startswith("# Task"):
            cleaned.append(l)
        elif stripped.startswith("#"):
            continue
        else:
            in_single = False
            in_double = False
            comment_start = -1
            for idx, ch in enumerate(l):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == '#' and not in_single and not in_double:
                    comment_start = idx
                    break
            if comment_start != -1:
                cleaned.append(l[:comment_start].rstrip())
            else:
                cleaned.append(l)

    final_lines = []
    prev_blank = False
    for l in cleaned:
        if l.strip() == '':
            if not prev_blank:
                final_lines.append('')
                prev_blank = True
        else:
            final_lines.append(l)
            prev_blank = False

    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in final_lines]
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

cells = [
    md_cell("# Assignment 11: Convolutional Neural Networks and Transfer Learning"),
    code_cell("""import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"""),

    md_cell("## Question 1: CNN Basics on a 3x3 Matrix"),
    code_cell("""np.random.seed(42)
torch.manual_seed(42)

matrix_3x3 = np.random.randn(3, 3)
print("Initial 3x3 Matrix:")
print(np.round(matrix_3x3, 4))"""),

    md_cell("### Convolution, Padding (P=1), and Activation Functions"),
    code_cell("""pad_p1 = np.pad(matrix_3x3, pad_width=1, mode='constant', constant_values=0)
print("Padded Matrix (5x5, P=1):")
print(np.round(pad_p1, 4))

kernel_3x3 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float64)
bias = 0.5

conv_out = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        conv_out[i, j] = np.sum(pad_p1[i:i+3, j:j+3] * kernel_3x3) + bias

print("Convolution Output (3x3):")
print(np.round(conv_out, 4))

relu_out = np.maximum(0, conv_out)
sigmoid_out = 1.0 / (1.0 + np.exp(-conv_out))

print("ReLU Activation:")
print(np.round(relu_out, 4))
print("Sigmoid Activation:")
print(np.round(sigmoid_out, 4))"""),

    md_cell("### Max Pooling, Min Pooling, and Global Average Pooling"),
    code_cell("""tensor_3x3 = torch.tensor(matrix_3x3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

max_pool = nn.MaxPool2d(kernel_size=2, stride=1)(tensor_3x3).squeeze().numpy()
min_pool = (-nn.MaxPool2d(kernel_size=2, stride=1)(-tensor_3x3)).squeeze().numpy()
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))(tensor_3x3).item()

print("Max Pooling (2x2, stride=1):")
print(np.round(max_pool, 4))
print("Min Pooling (2x2, stride=1):")
print(np.round(min_pool, 4))
print(f"Global Average Pooling: {global_avg_pool:.4f}")"""),

    md_cell("### Flattening, Dense Layers, Batch Normalization, Dropout, and Early Stopping"),
    code_cell("""flat_vector = tensor_3x3.view(1, -1)
print(f"Flattened Shape: {flat_vector.shape}")

batch_norm = nn.BatchNorm1d(num_features=9)
bn_out = batch_norm(flat_vector)
print("Batch Normalized Vector:")
print(np.round(bn_out.detach().numpy(), 4))

dense_layer = nn.Linear(in_features=9, out_features=4)
dense_out = dense_layer(bn_out)
print("Dense Output (4 units):")
print(np.round(dense_out.detach().numpy(), 4))

dropout = nn.Dropout(p=0.5)
dropout.train()
drop_out = dropout(dense_out)
print("Dropout Output (p=0.5):")
print(np.round(drop_out.detach().numpy(), 4))

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

es = EarlyStopping(patience=3)
test_losses = [1.5, 1.2, 1.22, 1.25, 1.3]
for ep, val_l in enumerate(test_losses):
    stopped = es(val_l)
    print(f"Epoch {ep+1}: Val Loss = {val_l:.2f}, Stopped = {stopped}")
    if stopped:
        break"""),

    md_cell("## Dataset Setup: 3 Classes x 200 Images (600 Images of 64x64x3)"),
    code_cell("""dataset_dir = 'tiny_dataset_600'
class_names = sorted(os.listdir(dataset_dir))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

image_paths = []
image_labels = []

for cls in class_names:
    files = glob.glob(os.path.join(dataset_dir, cls, '*.JPEG'))[:200]
    for f in files:
        image_paths.append(f)
        image_labels.append(class_to_idx[cls])

print(f"Loaded {len(image_paths)} images across classes: {class_to_idx}")

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, image_labels, test_size=0.30, random_state=42, stratify=image_labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    test_paths, test_labels, test_size=0.50, random_state=42, stratify=test_labels
)

print(f"Train set: {len(train_paths)} images (70%)")
print(f"Val set:   {len(val_paths)} images (15%)")
print(f"Test set:  {len(test_paths)} images (15%)")"""),

    code_cell("""class TinyImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(TinyImageDataset(train_paths, train_labels, transform_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TinyImageDataset(val_paths, val_labels, transform_test), batch_size=32, shuffle=False)
test_loader = DataLoader(TinyImageDataset(test_paths, test_labels, transform_test), batch_size=32, shuffle=False)

plt.figure(figsize=(9, 3))
for i in range(3):
    sample_cls_path = [p for p, l in zip(train_paths, train_labels) if l == i][0]
    sample_img = Image.open(sample_cls_path)
    plt.subplot(1, 3, i + 1)
    plt.imshow(sample_img)
    plt.title(f"Class {i}: {class_names[i]}\\nSize: {sample_img.size}")
    plt.axis('off')
plt.tight_layout()
plt.show()"""),

    md_cell("## Question 2: Custom CNN & Overfitting Analysis"),
    code_cell("""class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using compute device: {device}")"""),

    code_cell("""def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if early_stopper(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return train_losses, val_losses

custom_model = CustomCNN(num_classes=3).to(device)
train_losses, val_losses = train_model(custom_model, train_loader, val_loader, epochs=15, lr=0.001)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o', lw=2)
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s', lw=2)
plt.title('Custom CNN: Training vs Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()"""),

    md_cell("## Question 3: Transfer Learning Comparison"),
    code_cell("""def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

custom_metrics = evaluate_model(custom_model, test_loader)
print("Custom CNN Test Metrics:")
print(f"  Accuracy:  {custom_metrics[0]:.4f}")
print(f"  Precision: {custom_metrics[1]:.4f}")
print(f"  Recall:    {custom_metrics[2]:.4f}")
print(f"  F1 Score:  {custom_metrics[3]:.4f}")"""),

    code_cell("""vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
for param in vgg.features.parameters():
    param.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, 3)
vgg = vgg.to(device)
train_model(vgg, train_loader, val_loader, epochs=5, lr=0.0005)
vgg_metrics = evaluate_model(vgg, test_loader)

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in mobilenet.features.parameters():
    param.requires_grad = False
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 3)
mobilenet = mobilenet.to(device)
train_model(mobilenet, train_loader, val_loader, epochs=5, lr=0.0005)
mobilenet_metrics = evaluate_model(mobilenet, test_loader)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 3)
resnet = resnet.to(device)
train_model(resnet, train_loader, val_loader, epochs=5, lr=0.0005)
resnet_metrics = evaluate_model(resnet, test_loader)"""),

    code_cell("""models_list = ['Custom CNN', 'VGG16', 'MobileNetV2', 'ResNet18']
all_results = [custom_metrics, vgg_metrics, mobilenet_metrics, resnet_metrics]

header = f"{'Model':<14} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}"
print(header)
print("-" * len(header))
for name, m in zip(models_list, all_results):
    print(f"{name:<14} | {m[0]:<10.4f} | {m[1]:<10.4f} | {m[2]:<10.4f} | {m[3]:<10.4f}")

x = np.arange(len(models_list))
width = 0.2

plt.figure(figsize=(11, 5))
plt.bar(x - 1.5*width, [m[0] for m in all_results], width, label='Accuracy', color='royalblue')
plt.bar(x - 0.5*width, [m[1] for m in all_results], width, label='Precision', color='darkorange')
plt.bar(x + 0.5*width, [m[2] for m in all_results], width, label='Recall', color='forestgreen')
plt.bar(x + 1.5*width, [m[3] for m in all_results], width, label='F1 Score', color='firebrick')

plt.xlabel('Model Architecture')
plt.ylabel('Score (0 to 1)')
plt.title('Comparison of Custom CNN vs Transfer Learning Models (Test Set)')
plt.xticks(x, models_list)
plt.ylim([0, 1.1])
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()""")
]

with open('Image_Processing_Lab11.ipynb', 'w', encoding='utf-8') as f:
    json.dump(make_notebook(cells), f, indent=1)

print("Created Image_Processing_Lab11.ipynb successfully!")
