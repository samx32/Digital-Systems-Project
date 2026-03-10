# vgg16_cifar10.py — VGG-16 fine-tuned on CIFAR-10

"""
VGG-16 Baseline for CIFAR-10

Loads a pre-trained VGG-16 (ImageNet weights) and adapts it for
CIFAR-10's 32×32 images:
  - Replace avgpool with AdaptiveAvgPool2d(1, 1)
  - Replace classifier with 512 → 4096 → 4096 → 10 (matches 1×1 spatial)

Fine-tunes for 10 epochs with SGD + cosine annealing.

Usage:
    python -m src.baseline.vgg16_cifar10
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src.utils.energy_measurements import EnergyTracker


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_vgg16_cifar10(pretrained: bool = True) -> nn.Module:
    """
    Return a VGG-16 adapted for CIFAR-10 (32×32, 10 classes).

    With 32×32 input the feature extractor produces 512×1×1 after
    5 max-pool layers (32→16→8→4→2→1).  We replace the default
    AdaptiveAvgPool2d(7,7) with (1,1) and shrink the classifier
    input from 25088 to 512.
    """
    if pretrained:
        weights = models.VGG16_Weights.DEFAULT
    else:
        weights = None
    model = models.vgg16(weights=weights)

    # 32×32 → 512×1×1 after features, so use 1×1 pooling
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # New classifier: 512 → 4096 → 4096 → 10
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 10),
    )

    return model


def load_cifar10_data(batch_size: int = 64):
    """Load CIFAR-10 with standard augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root="./data", train=False,
                               download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return running_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return running_loss / len(loader), correct / total


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 10
    lr = 0.005

    print("\nLoading CIFAR-10...")
    # VGG-16 is memory-hungry — use smaller batch size
    train_loader, test_loader = load_cifar10_data(batch_size=64)

    print("Building VGG-16 (ImageNet pre-trained, adapted for CIFAR-10)...")
    model = get_vgg16_cifar10(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    # Lower LR for feature layers (pre-trained), higher for classifier (new)
    param_groups = [
        {"params": model.features.parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nFine-tuning for {num_epochs} epochs...")
    with EnergyTracker(experiment_name="vgg16_baseline") as tracker:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            print(f"  Epoch {epoch}/{num_epochs}  "
                  f"Train: {train_acc*100:.2f}%  Test: {test_acc*100:.2f}%")
        tracker.set_accuracy(test_acc)

    # Save
    save_dir = "./data/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vgg16_cifar10.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved VGG-16 to {save_path}")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print(f"Energy metrics: {tracker.metrics}")


if __name__ == "__main__":
    main()
