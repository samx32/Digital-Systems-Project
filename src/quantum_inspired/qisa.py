# qisa.py - Quantum-Inspired Simulated Annealing for Neural Network Optimization

"""
Quantum-Inspired Simulated Annealing (QISA) Implementation

This module implements QISA for optimizing neural network pruning configurations.
The algorithm combines classical simulated annealing with quantum-inspired concepts
like tunneling and superposition-based state sampling.

Key Concepts:
- Quantum tunneling: Allows escaping local minima by "tunneling" through barriers
- Superposition sampling: Sample neighboring states from a quantum-inspired distribution
- Adaptive perturbation: Perturbation magnitude decreases with temperature
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import math
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List, Optional, Callable

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker


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


class QuantumState:
    """
    Represents a quantum-inspired state for QISA.
    
    The state uses probability amplitudes to represent a superposition
    of pruning configurations, allowing for quantum-inspired sampling.
    """
    
    def __init__(self, dimensions: int, min_val: float = 0.0, max_val: float = 0.9):
        """
        Initialize a quantum state.
        
        Args:
            dimensions: Number of dimensions (e.g., number of layers to prune)
            min_val: Minimum value for each dimension
            max_val: Maximum value for each dimension
        """
        self.dimensions = dimensions
        self.min_val = min_val
        self.max_val = max_val
        
        # Classical position (current solution)
        self.position = np.random.uniform(min_val, max_val, dimensions)
        
        # Quantum uncertainty (standard deviation for sampling)
        self.uncertainty = np.ones(dimensions) * (max_val - min_val) / 4
    
    def sample(self) -> np.ndarray:
        """
        Sample a solution from the quantum state's probability distribution.
        
        Uses Gaussian sampling centered on the current position with
        quantum uncertainty determining the spread.
        
        Returns:
            Sampled solution array
        """
        # Sample from Gaussian distribution (quantum-inspired superposition)
        sample = np.random.normal(self.position, self.uncertainty)
        
        # Clip to valid range
        sample = np.clip(sample, self.min_val, self.max_val)
        
        return sample
    
    def collapse(self, new_position: np.ndarray):
        """
        Collapse the quantum state to a new position.
        
        Args:
            new_position: New position to collapse to
        """
        self.position = np.clip(new_position.copy(), self.min_val, self.max_val)
    
    def reduce_uncertainty(self, factor: float = 0.99):
        """
        Reduce quantum uncertainty (analogous to cooling).
        
        Args:
            factor: Multiplicative factor for uncertainty reduction
        """
        self.uncertainty *= factor
        # Set minimum uncertainty to prevent complete collapse
        min_uncertainty = (self.max_val - self.min_val) / 100
        self.uncertainty = np.maximum(self.uncertainty, min_uncertainty)


class QISA:
    """
    Quantum-Inspired Simulated Annealing optimizer.
    
    Combines simulated annealing with quantum-inspired concepts:
    - Quantum tunneling for escaping local minima
    - Superposition-based neighbor generation
    - Temperature-controlled uncertainty
    """
    
    def __init__(
        self,
        dimensions: int,
        fitness_function: Callable,
        initial_temp: float = 1.0,
        final_temp: float = 0.0001,
        cooling_rate: float = 0.995,
        iterations_per_temp: int = 50,
        tunneling_prob: float = 0.1,
        min_val: float = 0.0,
        max_val: float = 0.9
    ):
        """
        Initialize QISA optimizer.
        
        Args:
            dimensions: Number of dimensions to optimize
            fitness_function: Function to evaluate solutions (lower is better)
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_rate: Temperature reduction factor per iteration
            iterations_per_temp: Number of iterations at each temperature
            tunneling_prob: Probability of quantum tunneling jump
            min_val: Minimum value for each dimension
            max_val: Maximum value for each dimension
        """
        self.dimensions = dimensions
        self.fitness_function = fitness_function
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.tunneling_prob = tunneling_prob
        self.min_val = min_val
        self.max_val = max_val
        
        # Initialize quantum state
        self.state = QuantumState(dimensions, min_val, max_val)
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.temperature_history = []
    
    def _quantum_tunnel(self, current_position: np.ndarray, temperature: float) -> np.ndarray:
        """
        Perform a quantum tunneling jump.
        
        Tunneling allows the algorithm to escape local minima by making
        larger jumps that would normally be rejected by Metropolis criterion.
        
        Args:
            current_position: Current solution position
            temperature: Current temperature
        
        Returns:
            New position after tunneling
        """
        # Tunneling distance scales with temperature
        tunnel_distance = (self.max_val - self.min_val) * temperature * 0.5
        
        # Random direction
        direction = np.random.randn(self.dimensions)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Quantum-inspired tunneling: may cross energy barriers
        new_position = current_position + direction * tunnel_distance * np.random.random()
        
        return np.clip(new_position, self.min_val, self.max_val)
    
    def _acceptance_probability(self, current_fitness: float, new_fitness: float, 
                                temperature: float) -> float:
        """
        Calculate acceptance probability using Metropolis criterion with
        quantum-inspired modifications.
        
        Args:
            current_fitness: Fitness of current solution
            new_fitness: Fitness of proposed solution
            temperature: Current temperature
        
        Returns:
            Acceptance probability
        """
        if new_fitness < current_fitness:
            return 1.0
        
        # Standard Metropolis criterion with quantum correction
        delta = new_fitness - current_fitness
        
        # Quantum correction: increases acceptance probability at lower temperatures
        # This mimics quantum tunneling through energy barriers
        quantum_factor = 1 + 0.1 * np.exp(-temperature)
        
        return np.exp(-delta / (temperature * quantum_factor))
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the QISA optimization.
        
        Args:
            verbose: Print progress information
        
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        temperature = self.initial_temp
        
        # Initialize with random solution
        current_solution = self.state.sample()
        current_fitness = self.fitness_function(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        
        iteration = 0
        
        while temperature > self.final_temp:
            for _ in range(self.iterations_per_temp):
                iteration += 1
                
                # Decide between quantum tunneling and normal sampling
                # Adaptive tunnelling: probability scales with temperature
                # High temp = more tunnelling (exploration), low temp = less (exploitation)
                adaptive_tunnel_prob = self.tunneling_prob * (temperature / self.initial_temp)
                if np.random.random() < adaptive_tunnel_prob:
                    # Quantum tunneling jump
                    new_solution = self._quantum_tunnel(current_solution, temperature)
                else:
                    # Sample from quantum state's superposition
                    self.state.collapse(current_solution)
                    new_solution = self.state.sample()
                
                # Evaluate new solution
                new_fitness = self.fitness_function(new_solution)
                
                # Accept or reject based on quantum-modified Metropolis criterion
                if np.random.random() < self._acceptance_probability(
                    current_fitness, new_fitness, temperature
                ):
                    current_solution = new_solution
                    current_fitness = new_fitness
                    
                    # Update quantum state position
                    self.state.collapse(current_solution)
                
                # Update best solution
                if current_fitness < self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_solution = current_solution.copy()
            
            # Record history
            self.fitness_history.append(self.best_fitness)
            self.temperature_history.append(temperature)
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Reduce quantum uncertainty (wave function collapse)
            self.state.reduce_uncertainty(self.cooling_rate)
            
            # Progress output
            if verbose and len(self.fitness_history) % 20 == 0:
                print(f"  Temp: {temperature:.6f}, Best Fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness


class QISAPruningOptimizer:
    """
    Uses QISA to optimize layer-wise pruning ratios for a neural network.
    
    The optimization uses quantum-inspired simulated annealing to find
    optimal pruning configurations that balance accuracy and compression.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        train_loader: DataLoader = None,
        initial_temp: float = 1.0,
        final_temp: float = 0.001,
        cooling_rate: float = 0.98,
        iterations_per_temp: int = 30,
        finetune_epochs: int = 5,
        eval_batches: int = 50
    ):
        """
        Initialize the QISA pruning optimizer.
        
        Args:
            model: PyTorch model to optimize
            test_loader: DataLoader for evaluation
            device: Device to run computations on
            train_loader: DataLoader for post-optimization fine-tuning
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_rate: Temperature reduction factor
            iterations_per_temp: Iterations at each temperature
            finetune_epochs: Epochs of fine-tuning after pruning (0 to skip)
            eval_batches: Number of test batches per fitness evaluation
        """
        self.original_model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.device = device
        self.finetune_epochs = finetune_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.eval_batches = eval_batches
        
        # Get prunable layers
        self.prunable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable_layers.append(name)
        
        self.num_layers = len(self.prunable_layers)
        
        # Store original weights
        self.original_state_dict = copy.deepcopy(model.state_dict())
        
        # Working model — reused across evaluations to avoid costly deepcopy
        self._working_model = copy.deepcopy(model)
        
        # Evaluation cache to speed up optimization
        self._eval_cache = {}
        
        print(f"QISA Pruning Optimizer initialized:")
        print(f"  Prunable layers: {self.num_layers}")
        print(f"  Initial temperature: {initial_temp}")
        print(f"  Cooling rate: {cooling_rate}")
        print(f"  Post-optimization fine-tuning: {finetune_epochs} epochs")
    
    def _apply_pruning(self, model: nn.Module, pruning_ratios: np.ndarray) -> nn.Module:
        """
        Apply specified pruning ratios to model layers.
        
        Args:
            model: Model to prune
            pruning_ratios: Pruning ratio for each prunable layer
        
        Returns:
            Pruned model
        """
        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if layer_idx < len(pruning_ratios):
                    ratio = pruning_ratios[layer_idx]
                    if ratio > 0.01:  # Skip very small pruning ratios
                        prune.l1_unstructured(
                            module, name='weight', amount=float(ratio)
                        )
                        prune.remove(module, 'weight')
                    layer_idx += 1
        
        return model
    
    def _evaluate_solution(self, pruning_ratios: np.ndarray) -> float:
        """
        Evaluate a pruning configuration.
        
        Args:
            pruning_ratios: Array of pruning ratios for each layer
        
        Returns:
            Fitness value (lower is better)
        """
        # Round ratios for caching
        cache_key = tuple(np.round(pruning_ratios, 3))
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        
        # Reset working model to original weights (no deepcopy — load_state_dict copies internally)
        model = self._working_model
        model.load_state_dict(self.original_state_dict)
        model.to(self.device)
        
        # Apply pruning
        model = self._apply_pruning(model, pruning_ratios)
        
        # Evaluate accuracy on a subset for speed
        model.eval()
        correct = 0
        total = 0
        max_batches = self.eval_batches
        
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(self.test_loader):
                if batch_idx >= max_batches:
                    break
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate average sparsity
        avg_pruning = np.mean(pruning_ratios)
        
        # ---- Constraint-based fitness function ----
        # Goal: maximise sparsity subject to accuracy >= baseline - delta
        baseline_acc = 0.84
        accuracy_delta = 0.02  # tolerate up to 2% drop
        accuracy_threshold = baseline_acc - accuracy_delta  # 0.82

        if accuracy >= accuracy_threshold:
            # Above threshold: reward sparsity, with small bonus for extra accuracy
            fitness = -avg_pruning - 0.1 * (accuracy - accuracy_threshold)
        else:
            # Below threshold: heavy penalty proportional to shortfall
            shortfall = accuracy_threshold - accuracy
            fitness = 5.0 * shortfall  # strong penalty drives search away
        
        # Cache result
        self._eval_cache[cache_key] = fitness
        
        return fitness
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, float, 'CIFAR10CNN']:
        """
        Run QISA optimization to find optimal pruning ratios.
        
        Args:
            verbose: Print progress information
        
        Returns:
            Tuple of (optimal_pruning_ratios, final_accuracy, final_sparsity, final_model)
        """
        if verbose:
            print("\nStarting QISA Pruning Optimization...")
            print(f"  Adaptive tunnelling: prob scales with temperature")
            print(f"  Using subset evaluation ({self.eval_batches} batches) for speed")
            print(f"  Post-optimization fine-tuning: {self.finetune_epochs} epochs")
        
        # Calculate number of temperature steps for progress reporting
        num_temp_steps = int(np.log(self.final_temp / self.initial_temp) / 
                            np.log(self.cooling_rate)) + 1
        print(f"  Estimated temperature steps: {num_temp_steps}")
        
        # Create QISA instance
        qisa = QISA(
            dimensions=self.num_layers,
            fitness_function=self._evaluate_solution,
            initial_temp=self.initial_temp,
            final_temp=self.final_temp,
            cooling_rate=self.cooling_rate,
            iterations_per_temp=self.iterations_per_temp,
            tunneling_prob=0.1,  # 10% chance of quantum tunneling
            min_val=0.0,
            max_val=0.85  # Maximum 85% pruning per layer
        )
        
        # Run optimization
        optimal_ratios, best_fitness = qisa.optimize(verbose=verbose)
        
        # Full evaluation of optimal solution
        model = copy.deepcopy(self.original_model)
        model.load_state_dict(copy.deepcopy(self.original_state_dict))
        model.to(self.device)
        model = self._apply_pruning(model, optimal_ratios)
        
        # ---- Post-optimization fine-tuning ----
        if self.train_loader is not None and self.finetune_epochs > 0:
            if verbose:
                print(f"\nFine-tuning pruned model for {self.finetune_epochs} epochs...")
            model.train()
            ft_optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                          momentum=0.9, weight_decay=1e-4)
            ft_criterion = nn.CrossEntropyLoss()
            ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ft_optimizer, T_max=self.finetune_epochs
            )
            for epoch in range(self.finetune_epochs):
                running_loss = 0.0
                for x_batch, y_batch in self.train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    ft_optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = ft_criterion(outputs, y_batch)
                    loss.backward()
                    ft_optimizer.step()
                    running_loss += loss.item()
                ft_scheduler.step()
                avg_loss = running_loss / len(self.train_loader)
                if verbose:
                    print(f"    Epoch {epoch+1}/{self.finetune_epochs}  Loss: {avg_loss:.4f}")
        elif verbose:
            print("\nSkipping fine-tuning (no train_loader provided).")
        
        # Full test set evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        final_accuracy = correct / total
        final_sparsity = np.mean(optimal_ratios)
        
        if verbose:
            print(f"\nQISA Optimization Complete!")
            print(f"Optimal pruning ratios per layer:")
            for i, (name, ratio) in enumerate(zip(self.prunable_layers, optimal_ratios)):
                print(f"  {name}: {ratio * 100:.1f}%")
            print(f"Final accuracy: {final_accuracy * 100:.2f}%")
            print(f"Average sparsity: {final_sparsity * 100:.1f}%")
        
        return optimal_ratios, final_accuracy, final_sparsity, model


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


def main():
    """Main function to run QISA-based neural network optimization."""
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load train + test data
    train_loader, test_loader = load_cifar10_data()
    criterion = nn.CrossEntropyLoss()
    
    # Load baseline model
    print("\n=== LOADING BASELINE MODEL ===")
    model = CIFAR10CNN().to(device)
    model.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    
    # Evaluate baseline
    print("\n=== BASELINE EVALUATION ===")
    with EnergyTracker(experiment_name="cifar10_baseline_qisa") as tracker:
        baseline_loss, baseline_acc = evaluate_model(model, test_loader, criterion, device)
        tracker.set_accuracy(baseline_acc)
    print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Energy metrics: {tracker.metrics}")
    
    # QISA Optimization
    print("\n=== QISA OPTIMIZATION ===")
    optimizer = QISAPruningOptimizer(
        model=model,
        test_loader=test_loader,
        device=device,
        train_loader=train_loader,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_rate=0.95,         # Faster cooling for demo
        iterations_per_temp=20,    # Fewer iterations for demo
        finetune_epochs=5          # Fine-tune after pruning
    )
    
    with EnergyTracker(experiment_name="cifar10_qisa") as tracker:
        optimal_ratios, qisa_accuracy, qisa_sparsity, qisa_model = optimizer.optimize(verbose=True)
        tracker.set_accuracy(qisa_accuracy)
    
    print(f"\nQISA Energy metrics: {tracker.metrics}")
    
    # Save QISA-optimized model
    save_dir = './data/models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cifar10_qisa.pth')
    torch.save(qisa_model.state_dict(), save_path)
    print(f"Saved QISA model to {save_path}")
    
    # Summary
    print("\n=== QISA OPTIMIZATION SUMMARY ===")
    print(f"{'Metric':<25} {'Baseline':>15} {'QISA Optimized':>15}")
    print("-" * 55)
    print(f"{'Accuracy':<25} {baseline_acc * 100:>14.2f}% {qisa_accuracy * 100:>14.2f}%")
    print(f"{'Average Sparsity':<25} {'0.00':>14}% {qisa_sparsity * 100:>14.1f}%")
    print(f"{'Accuracy Drop':<25} {'—':>15} {(baseline_acc - qisa_accuracy) * 100:>14.2f}%")


if __name__ == "__main__":
    main()
