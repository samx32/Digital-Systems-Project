# cifar10_combined_optimization.py - Combined Pruning + Quantization

"""
Combined Pruning + Quantization Script

Applies unstructured pruning (at the best-performing level) followed by
dynamic and static INT8 quantization. This tests whether combining two
compression techniques yields better energy/size trade-offs than either alone.

Usage:
    python -m src.classical_optimisation.cifar10_combined_optimization
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.classical_optimisation.cifar10_quantization import (
    QuantizedCIFAR10CNN,
    apply_dynamic_quantization,
    apply_static_quantization,
    copy_weights_to_quantized,
    get_model_size,
)
from src.utils.energy_measurements import EnergyTracker


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRUNING_LEVELS = [0.2, 0.4, 0.6]   # Pruning ratios to combine with quantization
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 0.0001


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_data(batch_size=64):
    """Load CIFAR-10 train and test sets."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model accuracy and loss."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def prune_model(model, amount: float):
    """Apply L1 unstructured pruning to Conv2d and Linear layers."""
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model


def fine_tune(model, train_loader, criterion, optimizer, device, epochs=5):
    """Fine-tune after pruning."""
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        acc = correct / total
        avg = total_loss / len(train_loader)
        print(f"    Fine-tune [{epoch}/{epochs}] Loss: {avg:.4f}, Acc: {acc:.4f}")
    return model


def main():
    """Run combined pruning + quantization experiments."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = load_cifar10_data()
    criterion = nn.CrossEntropyLoss()
    save_dir = './data/models'
    os.makedirs(save_dir, exist_ok=True)

    # --- Baseline ---
    print("\n" + "=" * 65)
    print("BASELINE MODEL (FP32)")
    print("=" * 65)
    baseline = CIFAR10CNN().to(device)
    baseline.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    baseline_size = get_model_size(baseline)
    baseline_loss, baseline_acc = evaluate_model(baseline, test_loader, criterion, device)
    print(f"Accuracy: {baseline_acc * 100:.2f}%  |  Size: {baseline_size:.2f} MB")

    results = []

    for amount in PRUNING_LEVELS:
        tag = int(amount * 100)

        # ------------------------------------------------------------------
        # Step 1: Prune + fine-tune (on GPU)
        # ------------------------------------------------------------------
        print(f"\n{'=' * 65}")
        print(f"PRUNED {tag}% + QUANTIZATION")
        print(f"{'=' * 65}")

        model = CIFAR10CNN().to(device)
        model.load_state_dict(
            torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
        )
        model = prune_model(model, amount=amount)

        print(f"  Fine-tuning pruned-{tag}% model for {FINE_TUNE_EPOCHS} epochs...")
        opt = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        model = fine_tune(model, train_loader, criterion, opt, device, epochs=FINE_TUNE_EPOCHS)

        pruned_loss, pruned_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"  Pruned-{tag}% accuracy after fine-tune: {pruned_acc * 100:.2f}%")

        # Move to CPU for quantization
        model.cpu()
        model.eval()

        # ------------------------------------------------------------------
        # Step 2a: Pruned + Dynamic Quantization
        # ------------------------------------------------------------------
        print(f"\n  --- Pruned {tag}% + Dynamic Quantization ---")
        dyn_model = apply_dynamic_quantization(model)
        dyn_size = get_model_size(dyn_model)

        with EnergyTracker(experiment_name=f"cifar10_pruned{tag}_dynamic_quant") as tracker:
            dyn_loss, dyn_acc = evaluate_model(dyn_model, test_loader, criterion, torch.device('cpu'))
            tracker.set_accuracy(dyn_acc)

        print(f"  Accuracy: {dyn_acc * 100:.2f}%  |  Size: {dyn_size:.2f} MB  |  "
              f"Size reduction: {(1 - dyn_size / baseline_size) * 100:.1f}%")
        print(f"  Energy: {tracker.metrics}")

        # Save
        dyn_path = os.path.join(save_dir, f'cifar10_pruned_{tag}_dynamic_quantized.pth')
        torch.save(dyn_model, dyn_path)
        print(f"  Saved to {dyn_path}")

        results.append({
            'method': f'Pruned {tag}% + Dynamic Quant',
            'accuracy': dyn_acc,
            'size': dyn_size,
        })

        # ------------------------------------------------------------------
        # Step 2b: Pruned + Static Quantization
        # ------------------------------------------------------------------
        print(f"\n  --- Pruned {tag}% + Static Quantization ---")

        # Build QuantizedCIFAR10CNN, copy pruned weights, calibrate, convert
        q_model = QuantizedCIFAR10CNN()
        q_model = copy_weights_to_quantized(model, q_model)
        static_model = apply_static_quantization(q_model, train_loader, num_batches=100)
        static_size = get_model_size(static_model)

        with EnergyTracker(experiment_name=f"cifar10_pruned{tag}_static_quant") as tracker:
            static_loss, static_acc = evaluate_model(static_model, test_loader, criterion, torch.device('cpu'))
            tracker.set_accuracy(static_acc)

        print(f"  Accuracy: {static_acc * 100:.2f}%  |  Size: {static_size:.2f} MB  |  "
              f"Size reduction: {(1 - static_size / baseline_size) * 100:.1f}%")
        print(f"  Energy: {tracker.metrics}")

        # Save (state_dict — needs structure reconstruction at load time)
        static_path = os.path.join(save_dir, f'cifar10_pruned_{tag}_static_quantized.pth')
        torch.save(static_model.state_dict(), static_path)
        print(f"  Saved to {static_path}")

        results.append({
            'method': f'Pruned {tag}% + Static Quant',
            'accuracy': static_acc,
            'size': static_size,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("COMBINED OPTIMIZATION SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Method':<35} {'Accuracy':>10} {'Size (MB)':>10} {'Size Red.':>10}")
    print("-" * 65)
    print(f"{'Baseline (FP32)':<35} {baseline_acc * 100:>9.2f}% {baseline_size:>10.2f} {'—':>10}")
    for r in results:
        red = (1 - r['size'] / baseline_size) * 100
        print(f"{r['method']:<35} {r['accuracy'] * 100:>9.2f}% {r['size']:>10.2f} {red:>9.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
