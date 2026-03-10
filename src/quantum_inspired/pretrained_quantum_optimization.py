# pretrained_quantum_optimization.py — QIGA & QISA for pre-trained models

"""
Quantum-Inspired Optimisation for Pre-trained Models (ResNet-18, VGG-16)

Re-uses the existing QIGAPruningOptimizer and QISAPruningOptimizer with
the pre-trained models.  Both optimizers are model-agnostic — they
discover prunable Conv2d/Linear layers automatically.

Usage:
    python -m src.quantum_inspired.pretrained_quantum_optimization
"""

import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.resnet18_cifar10 import get_resnet18_cifar10
from src.baseline.vgg16_cifar10 import get_vgg16_cifar10
from src.quantum_inspired.qiga import QIGAPruningOptimizer
from src.quantum_inspired.qisa import QISAPruningOptimizer
from src.utils.energy_measurements import EnergyTracker


SAVE_DIR = "./data/models"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def run_qiga(model_name, model, train_loader, test_loader, device):
    """Run QIGA pruning optimizer on a model."""
    print(f"\n{'='*60}")
    print(f"  QIGA OPTIMIZATION: {model_name}")
    print(f"{'='*60}")

    qiga_opt = QIGAPruningOptimizer(
        model=model,
        test_loader=test_loader,
        device=device,
        train_loader=train_loader,
        bits_per_layer=4,       # 4 bits (16 levels) — sufficient for large models
        population_size=6,      # small pop — large models are slow to evaluate
        generations=10,         # fewer gens — pretrained start strong, converge fast
        finetune_epochs=2,
        eval_batches=15,        # 15 batches ≈ 1920 imgs — fast yet representative
    )

    with EnergyTracker(experiment_name=f"{model_name.lower()}_qiga") as tracker:
        ratios, acc, sparsity, optimized_model = qiga_opt.optimize(verbose=True)
        tracker.set_accuracy(acc)

    save_path = os.path.join(SAVE_DIR, f"{model_name.lower()}_qiga.pth")
    torch.save(optimized_model.state_dict(), save_path)
    print(f"\n  Saved to {save_path}")
    print(f"  Final accuracy: {acc*100:.2f}%  Sparsity: {sparsity*100:.1f}%")
    return acc, sparsity


def run_qisa(model_name, model, train_loader, test_loader, device):
    """Run QISA pruning optimizer on a model."""
    print(f"\n{'='*60}")
    print(f"  QISA OPTIMIZATION: {model_name}")
    print(f"{'='*60}")

    qisa_opt = QISAPruningOptimizer(
        model=model,
        test_loader=test_loader,
        device=device,
        train_loader=train_loader,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_rate=0.88,          # aggressive cooling → ~30 temp steps
        iterations_per_temp=8,      # 8 iters → ~240 total evals
        finetune_epochs=2,
        eval_batches=15,            # 15 batches ≈ 1920 imgs — fast yet representative
    )

    with EnergyTracker(experiment_name=f"{model_name.lower()}_qisa") as tracker:
        ratios, acc, sparsity, optimized_model = qisa_opt.optimize(verbose=True)
        tracker.set_accuracy(acc)

    save_path = os.path.join(SAVE_DIR, f"{model_name.lower()}_qisa.pth")
    torch.save(optimized_model.state_dict(), save_path)
    print(f"\n  Saved to {save_path}")
    print(f"  Final accuracy: {acc*100:.2f}%  Sparsity: {sparsity*100:.1f}%")
    return acc, sparsity


def optimize_model(model_name, model_fn, weight_path, train_loader, test_loader, device):
    """Run both QIGA and QISA on one model."""
    if not os.path.isfile(weight_path):
        print(f"[SKIP] {model_name} baseline not found at {weight_path}")
        return

    # QIGA
    model_qiga = model_fn(pretrained=False)
    model_qiga.load_state_dict(
        torch.load(weight_path, map_location="cpu", weights_only=True)
    )
    model_qiga.to(device)
    run_qiga(model_name, model_qiga, train_loader, test_loader, device)

    # QISA (fresh copy)
    model_qisa = model_fn(pretrained=False)
    model_qisa.load_state_dict(
        torch.load(weight_path, map_location="cpu", weights_only=True)
    )
    model_qisa.to(device)
    run_qisa(model_name, model_qisa, train_loader, test_loader, device)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader, test_loader = load_cifar10(batch_size=128)

    # --- ResNet-18 ---
    optimize_model(
        "ResNet18", get_resnet18_cifar10,
        os.path.join(SAVE_DIR, "resnet18_cifar10.pth"),
        train_loader, test_loader, device
    )

    # --- VGG-16 ---
    optimize_model(
        "VGG16", get_vgg16_cifar10,
        os.path.join(SAVE_DIR, "vgg16_cifar10.pth"),
        train_loader, test_loader, device
    )

    print("\n" + "=" * 60)
    print("  Quantum-inspired optimization of pre-trained models complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
