# pretrained_optimization.py — Classical optimizations for pre-trained models

"""
Classical Optimisation for Pre-trained Models (ResNet-18, VGG-16)

Applies the same optimization methods used on the custom CIFAR10CNN:
  1. Unstructured L1 pruning (20%, 40%, 60%) + fine-tune
  2. Dynamic INT8 quantization

Each optimization is run for both ResNet-18 and VGG-16.
All models are saved to data/models/ for the benchmark.

Usage:
    python -m src.classical_optimisation.pretrained_optimization
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.resnet18_cifar10 import get_resnet18_cifar10
from src.baseline.vgg16_cifar10 import get_vgg16_cifar10
from src.utils.energy_measurements import EnergyTracker


SAVE_DIR = "./data/models"
PRUNING_AMOUNTS = [0.2, 0.4, 0.6]
FINETUNE_EPOCHS = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cifar10(batch_size: int = 128):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return correct / total


def apply_unstructured_pruning(model, amount):
    """Apply global L1 unstructured pruning then make it permanent."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    # Make pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    return model


def finetune(model, train_loader, device, epochs=5, lr=0.001):
    """Quick fine-tune after pruning."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"      Finetune epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def run_pruning(model_name, model_fn, weight_path, train_loader, test_loader, device):
    """Prune a model at multiple levels, fine-tune, save."""
    print(f"\n{'='*60}")
    print(f"  PRUNING: {model_name}")
    print(f"{'='*60}")

    for amount in PRUNING_AMOUNTS:
        pct = int(amount * 100)
        print(f"\n  --- {model_name} Pruned {pct}% ---")

        # Load fresh baseline
        model = model_fn(pretrained=False)
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
        model.to(device)

        # Eval before
        acc_before = evaluate(model, test_loader, device)
        print(f"    Accuracy before pruning: {acc_before*100:.2f}%")

        # Prune
        with EnergyTracker(experiment_name=f"{model_name.lower()}_pruned_{pct}") as tracker:
            model = apply_unstructured_pruning(model, amount)
            acc_after_prune = evaluate(model, test_loader, device)
            print(f"    Accuracy after pruning:  {acc_after_prune*100:.2f}%")

            # Fine-tune
            print(f"    Fine-tuning for {FINETUNE_EPOCHS} epochs...")
            finetune(model, train_loader, device, epochs=FINETUNE_EPOCHS)
            acc_final = evaluate(model, test_loader, device)
            tracker.set_accuracy(acc_final)

        print(f"    Accuracy after fine-tune: {acc_final*100:.2f}%")

        # Save
        save_name = f"{model_name.lower()}_pruned_{pct}.pth"
        save_path = os.path.join(SAVE_DIR, save_name)
        torch.save(model.state_dict(), save_path)
        print(f"    Saved to {save_path}")


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------

def run_dynamic_quantization(model_name, model_fn, weight_path, test_loader, device):
    """Apply dynamic INT8 quantization, save."""
    print(f"\n{'='*60}")
    print(f"  DYNAMIC QUANTIZATION: {model_name}")
    print(f"{'='*60}")

    model = model_fn(pretrained=False)
    model.load_state_dict(
        torch.load(weight_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with EnergyTracker(experiment_name=f"{model_name.lower()}_dynamic_quant") as tracker:
        q_model = quant.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        q_model.eval()
        acc = evaluate(q_model, test_loader, torch.device("cpu"))
        tracker.set_accuracy(acc)

    print(f"  Accuracy: {acc*100:.2f}%")

    # Save full model (dynamic quant can't use state_dict easily)
    save_name = f"{model_name.lower()}_dynamic_quantized.pth"
    save_path = os.path.join(SAVE_DIR, save_name)
    torch.save(q_model, save_path)
    print(f"  Saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader, test_loader = load_cifar10(batch_size=128)

    # --- ResNet-18 ---
    resnet_path = os.path.join(SAVE_DIR, "resnet18_cifar10.pth")
    if os.path.isfile(resnet_path):
        run_pruning("ResNet18", get_resnet18_cifar10, resnet_path,
                    train_loader, test_loader, device)
        run_dynamic_quantization("ResNet18", get_resnet18_cifar10,
                                resnet_path, test_loader, device)
    else:
        print(f"[SKIP] ResNet-18 baseline not found at {resnet_path}")
        print("       Run: python -m src.baseline.resnet18_cifar10  first")

    # --- VGG-16 ---
    vgg_path = os.path.join(SAVE_DIR, "vgg16_cifar10.pth")
    if os.path.isfile(vgg_path):
        run_pruning("VGG16", get_vgg16_cifar10, vgg_path,
                    train_loader, test_loader, device)
        run_dynamic_quantization("VGG16", get_vgg16_cifar10,
                                vgg_path, test_loader, device)
    else:
        print(f"[SKIP] VGG-16 baseline not found at {vgg_path}")
        print("       Run: python -m src.baseline.vgg16_cifar10  first")

    print("\n" + "=" * 60)
    print("  Pre-trained model optimizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
