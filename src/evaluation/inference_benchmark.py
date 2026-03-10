# inference_benchmark.py - Standardised Inference Benchmark for All Models

"""
Inference Benchmark Script

Runs every saved model variant through an identical inference workload
(full CIFAR-10 test set, 10 000 images) and records:
  - accuracy
  - total inference time
  - average latency per batch
  - energy consumption (via EnergyTracker)
  - model size on disk

Results are saved to data/results/inference_benchmark.csv and printed
as a comparison table.

Usage:
    python -m src.evaluation.inference_benchmark
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.baseline.resnet18_cifar10 import get_resnet18_cifar10
from src.baseline.vgg16_cifar10 import get_vgg16_cifar10
from src.classical_optimisation.cifar10_quantization import QuantizedCIFAR10CNN  # needed for torch.load unpickling
from src.classical_optimisation.cifar10_structured_pruning import StructuredPrunedCNN  # needed for torch.load unpickling
from src.utils.energy_measurements import EnergyTracker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = "./data/models"
RESULTS_DIR = "./data/results"
BATCH_SIZE = 128
WARMUP_BATCHES = 5          # Warm-up batches (not measured)
NUM_RUNS = 10                # Repeat inference this many times and average

# Registry of models to benchmark.
# Each entry: (display_name, filename, loader_function_key)
# loader_function_key tells us how to reconstruct the model.
MODEL_REGISTRY = [
    # ---------- Custom CIFAR10CNN ----------
    ("Baseline (FP32)",         "cifar10_cnn.pth",                "baseline"),
    ("Pruned 20%",              "cifar10_pruned_20.pth",          "pruned"),
    ("Pruned 40%",              "cifar10_pruned_40.pth",          "pruned"),
    ("Pruned 60%",              "cifar10_pruned_60.pth",          "pruned"),
    ("Pruned 80%",              "cifar10_pruned_80.pth",          "pruned"),
    ("Struct Pruned 20%",       "cifar10_struct_pruned_20.pth",   "struct_pruned"),
    ("Struct Pruned 40%",       "cifar10_struct_pruned_40.pth",   "struct_pruned"),
    ("Struct Pruned 60%",       "cifar10_struct_pruned_60.pth",   "struct_pruned"),
    ("Dynamic Quantized INT8",  "cifar10_dynamic_quantized.pth",  "dynamic_quant"),
    ("Static Quantized INT8",   "cifar10_static_quantized.pth",   "static_quant"),
    ("Pruned 20% + Dyn Quant",  "cifar10_pruned_20_dynamic_quantized.pth",  "dynamic_quant"),
    ("Pruned 40% + Dyn Quant",  "cifar10_pruned_40_dynamic_quantized.pth",  "dynamic_quant"),
    ("Pruned 60% + Dyn Quant",  "cifar10_pruned_60_dynamic_quantized.pth",  "dynamic_quant"),
    ("Pruned 20% + Stat Quant", "cifar10_pruned_20_static_quantized.pth",   "static_quant"),
    ("Pruned 40% + Stat Quant", "cifar10_pruned_40_static_quantized.pth",   "static_quant"),
    ("Pruned 60% + Stat Quant", "cifar10_pruned_60_static_quantized.pth",   "static_quant"),
    ("QIGA Optimized",          "cifar10_qiga.pth",               "pruned"),
    ("QISA Optimized",          "cifar10_qisa.pth",               "pruned"),
    # ---------- ResNet-18 ----------
    ("RN18 Baseline",           "resnet18_cifar10.pth",           "resnet18"),
    ("RN18 Pruned 20%",         "resnet18_pruned_20.pth",         "resnet18"),
    ("RN18 Pruned 40%",         "resnet18_pruned_40.pth",         "resnet18"),
    ("RN18 Pruned 60%",         "resnet18_pruned_60.pth",         "resnet18"),
    ("RN18 Dyn Quant",          "resnet18_dynamic_quantized.pth", "resnet18_dyn_quant"),
    ("RN18 QIGA",               "resnet18_qiga.pth",              "resnet18"),
    ("RN18 QISA",               "resnet18_qisa.pth",              "resnet18"),
    # ---------- VGG-16 ----------
    ("VGG16 Baseline",          "vgg16_cifar10.pth",              "vgg16"),
    ("VGG16 Pruned 20%",        "vgg16_pruned_20.pth",            "vgg16"),
    ("VGG16 Pruned 40%",        "vgg16_pruned_40.pth",            "vgg16"),
    ("VGG16 Pruned 60%",        "vgg16_pruned_60.pth",            "vgg16"),
    ("VGG16 Dyn Quant",         "vgg16_dynamic_quantized.pth",    "vgg16_dyn_quant"),
    ("VGG16 QIGA",              "vgg16_qiga.pth",                 "vgg16"),
    ("VGG16 QISA",              "vgg16_qisa.pth",                 "vgg16"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cifar10_test(batch_size: int = 64) -> DataLoader:
    """Load the CIFAR-10 test set with standard normalisation."""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def load_model(name: str, filename: str, loader_key: str, device: torch.device):
    """
    Instantiate and load a model from disk.

    Returns:
        (model, device_to_use)  — quantized models must run on CPU.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.isfile(path):
        print(f"  [SKIP] {name}: {path} not found")
        return None, device

    if loader_key == "baseline" or loader_key == "pruned":
        # Standard CIFAR10CNN — pruned models have identical architecture
        # (pruning zeroes weights but keeps structure after prune.remove)
        model = CIFAR10CNN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.to(device)
        return model, device

    elif loader_key == "struct_pruned":
        # Structured pruned models have different architectures (fewer channels)
        # saved as full model via state_dict — need to load with weights_only=False
        # because the model class is StructuredPrunedCNN
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.to(device)
        model.eval()
        return model, device

    elif loader_key == "dynamic_quant":
        # Dynamic quantized model saved as full model (pickle)
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model, torch.device("cpu")

    elif loader_key == "static_quant":
        # Static quantized model saved as state_dict — reconstruct the
        # quantized model structure (fuse → prepare → convert) then load weights
        import torch.quantization as quant
        q_model = QuantizedCIFAR10CNN()
        q_model.eval()
        q_model.fuse_model()
        q_model.qconfig = quant.get_default_qconfig('x86')
        quant.prepare(q_model, inplace=True)
        quant.convert(q_model, inplace=True)
        q_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return q_model, torch.device("cpu")

    elif loader_key == "resnet18":
        # ResNet-18 (standard state_dict — baseline, pruned, or QIGA/QISA)
        model = get_resnet18_cifar10(pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.to(device)
        return model, device

    elif loader_key == "resnet18_dyn_quant":
        # ResNet-18 dynamic quantized (saved as full model)
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model, torch.device("cpu")

    elif loader_key == "vgg16":
        # VGG-16 (standard state_dict — baseline, pruned, or QIGA/QISA)
        model = get_vgg16_cifar10(pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.to(device)
        return model, device

    elif loader_key == "vgg16_dyn_quant":
        # VGG-16 dynamic quantized (saved as full model)
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model, torch.device("cpu")

    else:
        print(f"  [SKIP] {name}: unknown loader key '{loader_key}'")
        return None, device


def run_inference(model, dataloader, device, warmup_batches: int = 5):
    """
    Run a single inference pass over the full dataloader.

    Returns:
        (accuracy, total_time_seconds, per_batch_latencies)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Warm-up: run but don't measure
            if batch_idx < warmup_batches:
                _ = model(x_batch)
                continue

            start = time.perf_counter()
            outputs = model(x_batch)
            end = time.perf_counter()

            latencies.append(end - start)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    total_time = sum(latencies)
    return accuracy, total_time, latencies, total


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("  CIFAR-10 Inference Benchmark")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Default device: {device}\n")

    test_loader = load_cifar10_test(batch_size=BATCH_SIZE)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    for name, filename, loader_key in MODEL_REGISTRY:
        print(f"\n--- {name} ---")
        model, run_device = load_model(name, filename, loader_key, device)
        if model is None:
            continue

        filepath = os.path.join(MODELS_DIR, filename)
        size_mb = get_file_size_mb(filepath)
        print(f"  Model size: {size_mb:.2f} MB  |  Device: {run_device}")

        # If model runs on CPU, move test data there too
        if run_device != device:
            effective_loader = test_loader  # data is moved in run_inference
        else:
            effective_loader = test_loader

        all_accuracies = []
        all_times = []
        all_latencies = []
        all_images = []

        # Use EnergyTracker across all runs for this model
        experiment_tag = filename.replace(".pth", "").replace(" ", "_")
        with EnergyTracker(experiment_name=f"benchmark_{experiment_tag}") as tracker:
            for run_idx in range(NUM_RUNS):
                acc, total_time, latencies, num_images = run_inference(
                    model, effective_loader, run_device, warmup_batches=WARMUP_BATCHES
                )
                all_accuracies.append(acc)
                all_times.append(total_time)
                all_latencies.extend(latencies)
                all_images.append(num_images)
                throughput = num_images / total_time if total_time > 0 else 0
                print(f"  Run {run_idx + 1}/{NUM_RUNS}: acc={acc*100:.2f}%  time={total_time:.3f}s  throughput={throughput:.0f} img/s")
            tracker.set_accuracy(np.mean(all_accuracies))

        avg_acc = np.mean(all_accuracies)
        avg_time = np.mean(all_times)
        avg_latency_ms = np.mean(all_latencies) * 1000
        std_latency_ms = np.std(all_latencies) * 1000
        avg_images = np.mean(all_images)
        avg_throughput = avg_images / avg_time if avg_time > 0 else 0
        energy = tracker.metrics

        rows.append({
            "model": name,
            "accuracy_pct": round(avg_acc * 100, 2),
            "avg_inference_time_s": round(avg_time, 4),
            "avg_batch_latency_ms": round(avg_latency_ms, 3),
            "std_batch_latency_ms": round(std_latency_ms, 3),
            "throughput_imgs_per_s": round(avg_throughput, 1),
            "model_size_mb": round(size_mb, 2),
            "total_energy_joules": energy.get("total_energy_joules", 0),
            "cpu_energy_joules": energy.get("cpu_energy_joules", 0),
            "gpu_energy_joules": energy.get("gpu_energy_joules", 0),
            "emissions_gco2": energy.get("emissions_gco2", 0),
            "duration_seconds": energy.get("duration_seconds", 0),
        })

    # -----------------------------------------------------------------------
    # Save and display results
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "inference_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to {csv_path}")

    # Pretty-print comparison table
    print("\n" + "=" * 100)
    print("  INFERENCE BENCHMARK RESULTS")
    print("=" * 100)
    header = (
        f"{'Model':<26} {'Acc%':>7} {'Time(s)':>9} {'Lat(ms)':>9} "
        f"{'Img/s':>8} {'Size(MB)':>9} {'Energy(J)':>10} {'CO2(g)':>8}"
    )
    print(header)
    print("-" * 110)

    for _, row in df.iterrows():
        line = (
            f"{row['model']:<26} "
            f"{row['accuracy_pct']:>6.2f}% "
            f"{row['avg_inference_time_s']:>9.4f} "
            f"{row['avg_batch_latency_ms']:>9.3f} "
            f"{row['throughput_imgs_per_s']:>8.1f} "
            f"{row['model_size_mb']:>9.2f} "
            f"{row['total_energy_joules']:>10.4f} "
            f"{row['emissions_gco2']:>8.5f}"
        )
        print(line)

    print("=" * 110)
    print(f"Batch size: {BATCH_SIZE}  |  Warm-up: {WARMUP_BATCHES} batches  |  Runs: {NUM_RUNS}")
    print("=" * 100)


if __name__ == "__main__":
    main()
