# cifar10_quantum_optimization.py - Combined Quantum-Inspired Optimization for CIFAR-10

"""
Combined Quantum-Inspired Optimization Script

This script runs both QIGA and QISA optimization methods on the CIFAR-10 CNN model
and provides a comparative analysis of their performance.
"""

import numpy as np
import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker
from src.quantum_inspired.qiga import QIGAPruningOptimizer as QIGAOptimizer
from src.quantum_inspired.qisa import QISAPruningOptimizer as QISAOptimizer


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_data(batch_size: int = 64):
    """Load CIFAR-10 train and test sets."""
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
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


def count_nonzero_parameters(model):
    """Count non-zero parameters in the model."""
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()
    return total, nonzero


def main():
    """
    Main function to run combined quantum-inspired optimization experiments.
    
    This script:
    1. Evaluates the baseline model
    2. Runs QIGA optimization
    3. Runs QISA optimization
    4. Compares results from both methods
    """
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 70)
    
    # Load train + test data
    train_loader, test_loader = load_cifar10_data()
    criterion = nn.CrossEntropyLoss()
    
    # Results storage
    results = {}
    
    # ===========================================
    # BASELINE EVALUATION
    # ===========================================
    print("\n" + "=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)
    
    model = CIFAR10CNN().to(device)
    model.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    
    total_params, nonzero_params = count_nonzero_parameters(model)
    
    with EnergyTracker(experiment_name="cifar10_baseline_quantum") as tracker:
        baseline_loss, baseline_acc = evaluate_model(model, test_loader, criterion, device)
        tracker.set_accuracy(baseline_acc)
    
    results['baseline'] = {
        'accuracy': baseline_acc,
        'sparsity': 0.0,
        'energy': tracker.metrics,
        'total_params': total_params,
        'nonzero_params': nonzero_params
    }
    
    print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Total parameters: {total_params:,}")
    print(f"Energy metrics: {tracker.metrics}")
    
    # ===========================================
    # QIGA OPTIMIZATION
    # ===========================================
    print("\n" + "=" * 70)
    print("QIGA (QUANTUM-INSPIRED GENETIC ALGORITHM) OPTIMIZATION")
    print("=" * 70)
    
    # Reload fresh model
    model_qiga = CIFAR10CNN().to(device)
    model_qiga.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    
    qiga_optimizer = QIGAOptimizer(
        model=model_qiga,
        test_loader=test_loader,
        device=device,
        train_loader=train_loader,
        bits_per_layer=6,       # 64 pruning levels per layer (finer granularity)
        population_size=12,     # Smaller population for efficiency
        generations=25,         # Fewer generations for demo
        finetune_epochs=5       # Fine-tune after pruning
    )
    
    with EnergyTracker(experiment_name="cifar10_qiga") as tracker:
        qiga_ratios, qiga_acc, qiga_sparsity, qiga_model = qiga_optimizer.optimize(verbose=True)
        tracker.set_accuracy(qiga_acc)
    
    # Save QIGA model
    save_dir = './data/models'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(qiga_model.state_dict(), os.path.join(save_dir, 'cifar10_qiga.pth'))
    print(f"Saved QIGA model to {save_dir}/cifar10_qiga.pth")
    
    results['qiga'] = {
        'accuracy': qiga_acc,
        'sparsity': qiga_sparsity,
        'pruning_ratios': qiga_ratios,
        'energy': tracker.metrics
    }
    
    print(f"\nQIGA Final Results:")
    print(f"  Accuracy: {qiga_acc * 100:.2f}%")
    print(f"  Sparsity: {qiga_sparsity * 100:.1f}%")
    print(f"  Accuracy drop: {(baseline_acc - qiga_acc) * 100:.2f}%")
    
    # ===========================================
    # QISA OPTIMIZATION
    # ===========================================
    print("\n" + "=" * 70)
    print("QISA (QUANTUM-INSPIRED SIMULATED ANNEALING) OPTIMIZATION")
    print("=" * 70)
    
    # Reload fresh model
    model_qisa = CIFAR10CNN().to(device)
    model_qisa.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    
    qisa_optimizer = QISAOptimizer(
        model=model_qisa,
        test_loader=test_loader,
        device=device,
        train_loader=train_loader,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_rate=0.93,       # Adjusted for reasonable runtime
        iterations_per_temp=15,  # Fewer iterations for efficiency
        finetune_epochs=5        # Fine-tune after pruning
    )
    
    with EnergyTracker(experiment_name="cifar10_qisa") as tracker:
        qisa_ratios, qisa_acc, qisa_sparsity, qisa_model = qisa_optimizer.optimize(verbose=True)
        tracker.set_accuracy(qisa_acc)
    
    # Save QISA model
    torch.save(qisa_model.state_dict(), os.path.join(save_dir, 'cifar10_qisa.pth'))
    print(f"Saved QISA model to {save_dir}/cifar10_qisa.pth")
    
    results['qisa'] = {
        'accuracy': qisa_acc,
        'sparsity': qisa_sparsity,
        'pruning_ratios': qisa_ratios,
        'energy': tracker.metrics
    }
    
    print(f"\nQISA Final Results:")
    print(f"  Accuracy: {qisa_acc * 100:.2f}%")
    print(f"  Sparsity: {qisa_sparsity * 100:.1f}%")
    print(f"  Accuracy drop: {(baseline_acc - qisa_acc) * 100:.2f}%")
    
    # ===========================================
    # COMPARATIVE SUMMARY
    # ===========================================
    print("\n" + "=" * 70)
    print("QUANTUM-INSPIRED OPTIMIZATION COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Accuracy':>12} {'Sparsity':>12} {'Acc. Drop':>12}")
    print("-" * 60)
    print(f"{'Baseline':<20} {baseline_acc * 100:>11.2f}% {'0.00':>11}% {'—':>12}")
    print(f"{'QIGA':<20} {qiga_acc * 100:>11.2f}% {qiga_sparsity * 100:>11.1f}% {(baseline_acc - qiga_acc) * 100:>11.2f}%")
    print(f"{'QISA':<20} {qisa_acc * 100:>11.2f}% {qisa_sparsity * 100:>11.1f}% {(baseline_acc - qisa_acc) * 100:>11.2f}%")
    
    # Determine best method
    print("\n" + "-" * 60)
    
    # Calculate efficiency score: accuracy / (1 - accuracy_drop_ratio) * sparsity
    # Higher is better (want high accuracy and high sparsity)
    qiga_efficiency = qiga_acc * (1 + qiga_sparsity)
    qisa_efficiency = qisa_acc * (1 + qisa_sparsity)
    
    if qiga_efficiency > qisa_efficiency:
        print(f"QIGA achieves better efficiency score: {qiga_efficiency:.4f} vs {qisa_efficiency:.4f}")
    elif qisa_efficiency > qiga_efficiency:
        print(f"QISA achieves better efficiency score: {qisa_efficiency:.4f} vs {qiga_efficiency:.4f}")
    else:
        print(f"Both methods achieve similar efficiency: {qiga_efficiency:.4f}")
    
    # Energy comparison
    print("\nEnergy Consumption Comparison:")
    qiga_duration = results['qiga']['energy'].get('duration_seconds', 0)
    qisa_duration = results['qisa']['energy'].get('duration_seconds', 0)
    baseline_duration = results['baseline']['energy'].get('duration_seconds', 0)
    
    print(f"  Baseline inference: {baseline_duration:.2f}s")
    print(f"  QIGA optimization:  {qiga_duration:.2f}s")
    print(f"  QISA optimization:  {qisa_duration:.2f}s")
    
    print("\n" + "=" * 70)
    print("Optimization complete! Results saved to data/results/")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
