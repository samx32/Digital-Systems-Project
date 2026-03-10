# cifar10_quantization.py

import os

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant
import copy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_data(batch_size=64):
    """Load CIFAR-10 train and test sets with standard transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model and return loss + accuracy."""
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


def get_model_size(model):
    """Get model size in MB by saving to a buffer."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 ** 2)
    return size_mb


# -- Quantization-ready version of the CNN --
# PyTorch quantization needs QuantStub/DeQuantStub to mark where
# tensors go from floating point to quantized and back
class QuantizedCIFAR10CNN(nn.Module):
    def __init__(self):
        super(QuantizedCIFAR10CNN, self).__init__()

        # These stubs tell PyTorch where to convert float <-> quantized
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # Same architecture as the baseline CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.quant(x)       # float -> quantized
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = self.dequant(x)     # quantized -> float
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU layers for better quantization performance.
        This merges operations that are done separately into single fused ops."""

        # Fuse each Conv -> BatchNorm -> ReLU group in the conv layers
        # Indices correspond to positions in self.conv_layers Sequential
        torch.quantization.fuse_modules(self.conv_layers, ["0", "1", "2"], inplace=True)   # Conv block 1a
        torch.quantization.fuse_modules(self.conv_layers, ["3", "4", "5"], inplace=True)   # Conv block 1b
        torch.quantization.fuse_modules(self.conv_layers, ["8", "9", "10"], inplace=True)  # Conv block 2a
        torch.quantization.fuse_modules(self.conv_layers, ["11", "12", "13"], inplace=True) # Conv block 2b
        torch.quantization.fuse_modules(self.conv_layers, ["16", "17", "18"], inplace=True) # Conv block 3a
        torch.quantization.fuse_modules(self.conv_layers, ["19", "20", "21"], inplace=True) # Conv block 3b


def copy_weights_to_quantized(original_model, quantized_model):
    """Copy weights from the baseline model into the quantized version."""
    orig_state = original_model.state_dict()
    quant_state = quantized_model.state_dict()

    # Map matching keys (skip quant/dequant stub keys)
    for key in orig_state:
        if key in quant_state:
            quant_state[key] = orig_state[key]

    quantized_model.load_state_dict(quant_state)
    return quantized_model


def apply_dynamic_quantization(model):
    """Apply dynamic quantization - quantizes weights ahead of time,
    activations are quantized on-the-fly during inference.
    Simplest form, only targets Linear layers."""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},       # which layer types to quantize
        dtype=torch.qint8  # use 8-bit integers
    )
    return quantized_model


def apply_static_quantization(model, calibration_loader, num_batches=100):
    """Apply static quantization - needs a calibration step where we run
    sample data through the model so it can learn the right scale/zero-point
    for each layer's activations."""

    model.eval()
    model.cpu()

    # Fuse layers first for better quantization results
    model.fuse_model()

    # Set the quantization config (how weights and activations get quantized)
    model.qconfig = quant.get_default_qconfig('x86')

    # Insert observers that will record activation ranges during calibration
    quant.prepare(model, inplace=True)

    # Calibration pass - run sample data so observers can collect stats
    print(f"  Calibrating with {num_batches} batches...")
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            model(x_batch.cpu())

    # Convert the model to use quantized operations based on the calibration data
    quant.convert(model, inplace=True)

    return model


def main():
    """Main function to quantize the model, evaluate, and measure energy."""
    set_seed(42)

    # Quantization only works on CPU in PyTorch
    device = torch.device('cpu')
    print(f"Using device: {device} (quantization requires CPU)")

    # Load data
    train_loader, test_loader = load_cifar10_data()
    criterion = nn.CrossEntropyLoss()

    # --- Baseline evaluation ---
    print("\n=== BASELINE MODEL (FP32) ===")
    baseline_model = CIFAR10CNN().to(device)
    baseline_model.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )

    baseline_size = get_model_size(baseline_model)
    print(f"Model size: {baseline_size:.2f} MB")

    with EnergyTracker(experiment_name="cifar10_baseline") as tracker:
        baseline_loss, baseline_acc = evaluate_model(baseline_model, test_loader, criterion, device)
        tracker.set_accuracy(baseline_acc)

    print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Baseline energy: {tracker.metrics}")

    # --- Dynamic Quantization (INT8) ---
    print("\n=== DYNAMIC QUANTIZATION (INT8) ===")

    # Reload fresh model and apply dynamic quantization
    dyn_model = CIFAR10CNN().to(device)
    dyn_model.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    dyn_model = apply_dynamic_quantization(dyn_model)

    dyn_size = get_model_size(dyn_model)
    print(f"Model size: {dyn_size:.2f} MB (reduction: {(1 - dyn_size / baseline_size) * 100:.1f}%)")

    with EnergyTracker(experiment_name="cifar10_dynamic_quantized") as tracker:
        dyn_loss, dyn_acc = evaluate_model(dyn_model, test_loader, criterion, device)
        tracker.set_accuracy(dyn_acc)

    print(f"Dynamic quantized accuracy: {dyn_acc * 100:.2f}%")
    print(f"Accuracy drop from baseline: {(baseline_acc - dyn_acc) * 100:.2f}%")
    print(f"Energy metrics: {tracker.metrics}")

    # Save dynamic quantized model (full model, not state_dict,
    # because quantized layers use packed parameters)
    save_dir = './data/models'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(dyn_model, os.path.join(save_dir, 'cifar10_dynamic_quantized.pth'))
    print(f"Saved dynamic quantized model to {save_dir}/cifar10_dynamic_quantized.pth")

    # --- Static Quantization (INT8) ---
    print("\n=== STATIC QUANTIZATION (INT8) ===")

    # Build the quantization-aware model and copy weights from baseline
    static_model = QuantizedCIFAR10CNN()
    static_model = copy_weights_to_quantized(baseline_model, static_model)

    # Apply static quantization with calibration on training data
    static_model = apply_static_quantization(static_model, train_loader, num_batches=100)

    static_size = get_model_size(static_model)
    print(f"Model size: {static_size:.2f} MB (reduction: {(1 - static_size / baseline_size) * 100:.1f}%)")

    with EnergyTracker(experiment_name="cifar10_static_quantized") as tracker:
        static_loss, static_acc = evaluate_model(static_model, test_loader, criterion, device)
        tracker.set_accuracy(static_acc)

    print(f"Static quantized accuracy: {static_acc * 100:.2f}%")
    print(f"Accuracy drop from baseline: {(baseline_acc - static_acc) * 100:.2f}%")
    print(f"Energy metrics: {tracker.metrics}")

    # Save static quantized model (state_dict only — the model structure
    # must be reconstructed at load time via fuse+prepare+convert)
    torch.save(static_model.state_dict(), os.path.join(save_dir, 'cifar10_static_quantized.pth'))
    print(f"Saved static quantized model to {save_dir}/cifar10_static_quantized.pth")

    # --- Summary ---
    print("\n=== QUANTIZATION SUMMARY ===")
    print(f"{'Method':<25} {'Accuracy':>10} {'Size (MB)':>10} {'Size Reduction':>15}")
    print("-" * 65)
    print(f"{'Baseline (FP32)':<25} {baseline_acc * 100:>9.2f}% {baseline_size:>10.2f} {'—':>15}")
    print(f"{'Dynamic Quant (INT8)':<25} {dyn_acc * 100:>9.2f}% {dyn_size:>10.2f} {(1 - dyn_size / baseline_size) * 100:>14.1f}%")
    print(f"{'Static Quant (INT8)':<25} {static_acc * 100:>9.2f}% {static_size:>10.2f} {(1 - static_size / baseline_size) * 100:>14.1f}%")


if __name__ == "__main__":
    main()
