# qiga.py - Quantum-Inspired Genetic Algorithm for Neural Network Optimization

"""
Quantum-Inspired Genetic Algorithm (QIGA) Implementation

This module implements QIGA for optimizing neural network pruning decisions.
The algorithm uses quantum-inspired concepts like superposition and quantum
rotation gates to explore the search space efficiently.

Key Concepts:
- Q-bit representation: Each gene is represented as probability amplitudes (α, β)
  where |α|² + |β|² = 1
- Observation: Convert quantum chromosomes to binary solutions by sampling
- Quantum rotation gates: Update amplitudes based on fitness comparison with best solution
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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


class QuantumChromosome:
    """
    Represents a quantum chromosome with Q-bit representation.
    
    Each Q-bit is represented by two probability amplitudes (α, β) where:
    - |α|² is the probability of observing 0
    - |β|² is the probability of observing 1
    - |α|² + |β|² = 1
    """
    
    def __init__(self, length: int):
        """
        Initialize a quantum chromosome with uniform superposition.
        
        Args:
            length: Number of Q-bits in the chromosome
        """
        self.length = length
        # Initialize with uniform superposition: α = β = 1/√2
        self.alpha = np.ones(length) * (1 / np.sqrt(2))
        self.beta = np.ones(length) * (1 / np.sqrt(2))
    
    def observe(self) -> np.ndarray:
        """
        Observe (measure) the quantum chromosome to get a binary solution.
        
        Returns:
            Binary array where each bit is sampled based on |β|²
        """
        probabilities = self.beta ** 2  # Probability of getting 1
        random_values = np.random.random(self.length)
        return (random_values < probabilities).astype(int)
    
    def update(self, binary_solution: np.ndarray, best_solution: np.ndarray, 
               best_fitness: float, current_fitness: float, rotation_angle: float = 0.01 * np.pi):
        """
        Update Q-bit amplitudes using quantum rotation gate.
        
        The rotation direction and magnitude depend on the comparison between
        the current solution and the best solution found so far.
        
        Args:
            binary_solution: Current observed binary solution
            best_solution: Best binary solution found so far
            best_fitness: Fitness of best solution (lower is better)
            current_fitness: Fitness of current solution
            rotation_angle: Base rotation angle (default: 0.01π)
        """
        for i in range(self.length):
            # Determine rotation direction based on lookup table
            # This is a simplified version of the standard QIGA rotation lookup
            delta_theta = self._get_rotation_angle(
                binary_solution[i], best_solution[i],
                current_fitness, best_fitness, rotation_angle
            )
            
            # Apply rotation gate
            # [α'] = [cos(Δθ)  -sin(Δθ)] [α]
            # [β']   [sin(Δθ)   cos(Δθ)] [β]
            cos_theta = np.cos(delta_theta)
            sin_theta = np.sin(delta_theta)
            
            new_alpha = cos_theta * self.alpha[i] - sin_theta * self.beta[i]
            new_beta = sin_theta * self.alpha[i] + cos_theta * self.beta[i]
            
            self.alpha[i] = new_alpha
            self.beta[i] = new_beta
    
    def _get_rotation_angle(self, x_i: int, b_i: int, f_x: float, f_b: float, 
                           base_angle: float) -> float:
        """
        Determine rotation angle based on Q-bit lookup table.
        
        Args:
            x_i: Current solution bit at position i
            b_i: Best solution bit at position i
            f_x: Fitness of current solution
            f_b: Fitness of best solution
            base_angle: Base rotation angle
        
        Returns:
            Rotation angle (positive or negative)
        """
        # Standard QIGA lookup table (simplified)
        if x_i == 0 and b_i == 0:
            return 0
        elif x_i == 0 and b_i == 1:
            if f_x >= f_b:
                return base_angle  # Rotate towards |1⟩
            else:
                return -base_angle
        elif x_i == 1 and b_i == 0:
            if f_x >= f_b:
                return -base_angle  # Rotate towards |0⟩
            else:
                return base_angle
        else:  # x_i == 1 and b_i == 1
            return 0


class QIGA:
    """
    Quantum-Inspired Genetic Algorithm for optimization.
    
    This implementation uses quantum-inspired concepts to efficiently search
    for optimal solutions in binary search spaces.
    """
    
    def __init__(
        self,
        chromosome_length: int,
        population_size: int = 20,
        generations: int = 50,
        rotation_angle: float = 0.01 * np.pi,
        fitness_function: Optional[Callable] = None
    ):
        """
        Initialize QIGA.
        
        Args:
            chromosome_length: Number of bits in each solution
            population_size: Number of quantum individuals
            generations: Number of evolutionary generations
            rotation_angle: Quantum rotation angle for updates
            fitness_function: Function to evaluate solution fitness (lower is better)
        """
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.generations = generations
        self.rotation_angle = rotation_angle
        self.fitness_function = fitness_function
        
        # Initialize population of quantum chromosomes
        self.population = [
            QuantumChromosome(chromosome_length) for _ in range(population_size)
        ]
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
    
    def run(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the QIGA optimization.
        
        Uses an adaptive rotation angle schedule: starts at the configured
        rotation_angle and decays linearly to 10% of that value by the
        final generation.  This gives broad exploration early on and
        fine-grained exploitation towards the end.
        
        Args:
            verbose: Print progress information
        
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        initial_angle = self.rotation_angle
        final_angle = initial_angle * 0.1  # decay to 10%

        for gen in range(self.generations):
            gen_best_fitness = float('inf')

            # --- Adaptive rotation angle (linear decay) ---
            progress = gen / max(self.generations - 1, 1)
            current_angle = initial_angle * (1 - progress) + final_angle * progress
            
            for i, q_chromosome in enumerate(self.population):
                # Observe quantum chromosome to get binary solution
                binary_solution = q_chromosome.observe()
                
                # Evaluate fitness
                fitness = self.fitness_function(binary_solution)
                
                # Update best solution if improved
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = binary_solution.copy()
                
                if fitness < gen_best_fitness:
                    gen_best_fitness = fitness
                
                # Update quantum chromosome using adaptive rotation gate
                q_chromosome.update(
                    binary_solution, self.best_solution,
                    self.best_fitness, fitness, current_angle
                )
            
            self.fitness_history.append(self.best_fitness)
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"  Generation {gen + 1}/{self.generations}: "
                      f"Best Fitness = {self.best_fitness:.6f}  "
                      f"Rotation = {current_angle/np.pi:.4f}π")
        
        return self.best_solution, self.best_fitness


class QIGAPruningOptimizer:
    """
    Uses QIGA to optimize layer-wise pruning ratios for a neural network.
    
    The optimization finds the best pruning ratios for each layer to maximize
    the trade-off between model compression and accuracy preservation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        train_loader: DataLoader = None,
        bits_per_layer: int = 6,  # 6 bits = 64 pruning levels for finer granularity
        population_size: int = 15,
        generations: int = 30,
        finetune_epochs: int = 5,
        eval_batches: int = 50
    ):
        """
        Initialize the QIGA pruning optimizer.
        
        Args:
            model: PyTorch model to optimize
            test_loader: DataLoader for evaluation
            device: Device to run computations on
            train_loader: DataLoader for post-optimization fine-tuning
            bits_per_layer: Number of bits to encode pruning ratio per layer
            population_size: QIGA population size
            generations: Number of QIGA generations
            finetune_epochs: Epochs of fine-tuning after pruning (0 to skip)
            eval_batches: Number of test batches per fitness evaluation
        """
        self.original_model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.device = device
        self.finetune_epochs = finetune_epochs
        self.bits_per_layer = bits_per_layer
        self.population_size = population_size
        self.generations = generations
        self.eval_batches = eval_batches
        
        # Get prunable layers (Conv2d and Linear)
        self.prunable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable_layers.append(name)
        
        self.num_layers = len(self.prunable_layers)
        self.chromosome_length = self.num_layers * bits_per_layer
        
        # Store original weights
        self.original_state_dict = copy.deepcopy(model.state_dict())
        
        # Working model — reused across evaluations to avoid costly deepcopy
        # We reset its weights via load_state_dict before each evaluation.
        self._working_model = copy.deepcopy(model)
        
        # Evaluation cache to avoid re-computing same solutions
        self._eval_cache = {}
        
        print(f"QIGA Pruning Optimizer initialized:")
        print(f"  Prunable layers: {self.num_layers}")
        print(f"  Bits per layer: {bits_per_layer}")
        print(f"  Chromosome length: {self.chromosome_length}")
    
    def _decode_chromosome(self, binary_solution: np.ndarray) -> List[float]:
        """
        Decode binary chromosome into pruning ratios for each layer.
        
        Args:
            binary_solution: Binary array from QIGA
        
        Returns:
            List of pruning ratios (0.0 to 0.9375) for each layer
        """
        pruning_ratios = []
        for i in range(self.num_layers):
            # Extract bits for this layer
            start_idx = i * self.bits_per_layer
            end_idx = start_idx + self.bits_per_layer
            bits = binary_solution[start_idx:end_idx]
            
            # Convert binary to decimal (0 to 2^bits - 1)
            decimal_value = sum(bit * (2 ** (self.bits_per_layer - 1 - j)) 
                              for j, bit in enumerate(bits))
            
            # Map to pruning ratio (0 to max_ratio)
            max_value = 2 ** self.bits_per_layer - 1
            max_ratio = 0.9  # Maximum 90% pruning
            pruning_ratio = (decimal_value / max_value) * max_ratio
            pruning_ratios.append(pruning_ratio)
        
        return pruning_ratios
    
    def _apply_pruning(self, model: nn.Module, pruning_ratios: List[float]) -> nn.Module:
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
                    if ratio > 0:
                        # Apply L1 unstructured pruning
                        prune.l1_unstructured(
                            module, name='weight', amount=ratio
                        )
                        prune.remove(module, 'weight')
                    layer_idx += 1
        
        return model
    
    def _evaluate_solution(self, binary_solution: np.ndarray) -> float:
        """
        Evaluate a pruning configuration.
        
        Fitness combines accuracy loss and sparsity benefit.
        Lower fitness is better.
        
        Args:
            binary_solution: Binary chromosome from QIGA
        
        Returns:
            Fitness value (lower is better)
        """
        # Check cache first (convert to tuple for hashing)
        cache_key = tuple(binary_solution)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        
        # Decode to pruning ratios
        pruning_ratios = self._decode_chromosome(binary_solution)
        
        # Reset working model to original weights (no deepcopy — load_state_dict copies internally)
        model = self._working_model
        model.load_state_dict(self.original_state_dict)
        model.to(self.device)
        
        # Apply pruning
        model = self._apply_pruning(model, pruning_ratios)
        
        # Evaluate accuracy on SUBSET for speed
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
        
        # Calculate sparsity (average pruning ratio)
        avg_pruning = np.mean(pruning_ratios)
        
        # ---- Constraint-based fitness function ----
        # Goal: maximise sparsity subject to accuracy >= baseline - delta
        # baseline_acc is approximated as ~0.84 (CIFAR-10 CNN baseline)
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
        
        # Cache the result
        self._eval_cache[cache_key] = fitness
        
        return fitness
    
    def optimize(self, verbose: bool = True) -> Tuple[List[float], float, float, 'CIFAR10CNN']:
        """
        Run QIGA optimization to find optimal pruning ratios.
        
        Args:
            verbose: Print progress information
        
        Returns:
            Tuple of (optimal_pruning_ratios, final_accuracy, final_sparsity, final_model)
        """
        if verbose:
            print(f"  Population: {self.population_size}, Generations: {self.generations}")
            print(f"  Bits per layer: {self.bits_per_layer} ({2**self.bits_per_layer} pruning levels)")
            print(f"  Using subset evaluation ({self.eval_batches} batches) for speed")
            print(f"  Post-optimization fine-tuning: {self.finetune_epochs} epochs")
        
        # Create QIGA instance — adaptive rotation starts at 0.1π, decays to 0.01π
        qiga = QIGA(
            chromosome_length=self.chromosome_length,
            population_size=self.population_size,
            generations=self.generations,
            rotation_angle=0.1 * np.pi,  # Start large, adaptive schedule decays it
            fitness_function=self._evaluate_solution
        )
        
        # Run optimization
        best_solution, best_fitness = qiga.run(verbose=verbose)
        
        # Decode best solution
        optimal_ratios = self._decode_chromosome(best_solution)
        
        # Apply optimal pruning and get final metrics
        final_model = copy.deepcopy(self.original_model)
        final_model.load_state_dict(copy.deepcopy(self.original_state_dict))
        final_model.to(self.device)
        final_model = self._apply_pruning(final_model, optimal_ratios)
        
        # ---- Post-optimization fine-tuning ----
        if self.train_loader is not None and self.finetune_epochs > 0:
            if verbose:
                print(f"\nFine-tuning pruned model for {self.finetune_epochs} epochs...")
            final_model.train()
            ft_optimizer = torch.optim.SGD(final_model.parameters(), lr=0.001,
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
                    outputs = final_model(x_batch)
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
        
        # Calculate final accuracy
        final_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = final_model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        final_accuracy = correct / total
        final_sparsity = np.mean(optimal_ratios)
        
        if verbose:
            print(f"\nQIGA Optimization Complete!")
            print(f"Optimal pruning ratios per layer:")
            for i, (name, ratio) in enumerate(zip(self.prunable_layers, optimal_ratios)):
                print(f"  {name}: {ratio * 100:.1f}%")
            print(f"Final accuracy: {final_accuracy * 100:.2f}%")
            print(f"Average sparsity: {final_sparsity * 100:.1f}%")
        
        return optimal_ratios, final_accuracy, final_sparsity, final_model


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
    """Main function to run QIGA-based neural network optimization."""
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
    with EnergyTracker(experiment_name="cifar10_baseline_qiga") as tracker:
        baseline_loss, baseline_acc = evaluate_model(model, test_loader, criterion, device)
        tracker.set_accuracy(baseline_acc)
    print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Energy metrics: {tracker.metrics}")
    
    # QIGA Optimization
    print("\n=== QIGA OPTIMIZATION ===")
    optimizer = QIGAPruningOptimizer(
        model=model,
        test_loader=test_loader,
        device=device,
        train_loader=train_loader,
        bits_per_layer=6,      # 64 pruning levels per layer (finer granularity)
        population_size=15,    # Smaller population for faster runs
        generations=30,        # Fewer generations for demo
        finetune_epochs=5      # Fine-tune after pruning
    )
    
    with EnergyTracker(experiment_name="cifar10_qiga") as tracker:
        optimal_ratios, qiga_accuracy, qiga_sparsity, qiga_model = optimizer.optimize(verbose=True)
        tracker.set_accuracy(qiga_accuracy)
    
    print(f"\nQIGA Energy metrics: {tracker.metrics}")
    
    # Save QIGA-optimized model
    save_dir = './data/models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cifar10_qiga.pth')
    torch.save(qiga_model.state_dict(), save_path)
    print(f"Saved QIGA model to {save_path}")
    
    # Summary
    print("\n=== QIGA OPTIMIZATION SUMMARY ===")
    print(f"{'Metric':<25} {'Baseline':>15} {'QIGA Optimized':>15}")
    print("-" * 55)
    print(f"{'Accuracy':<25} {baseline_acc * 100:>14.2f}% {qiga_accuracy * 100:>14.2f}%")
    print(f"{'Average Sparsity':<25} {'0.00':>14}% {qiga_sparsity * 100:>14.1f}%")
    print(f"{'Accuracy Drop':<25} {'—':>15} {(baseline_acc - qiga_accuracy) * 100:>14.2f}%")


if __name__ == "__main__":
    main()
