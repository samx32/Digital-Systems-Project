# quantum_inspired/__init__.py

"""
Quantum-Inspired Optimization Algorithms

This module contains implementations of quantum-inspired optimization algorithms
for neural network pruning and optimization.

Available algorithms:
- QIGA: Quantum-Inspired Genetic Algorithm
- QISA: Quantum-Inspired Simulated Annealing
"""

from .qiga import QIGA, QuantumChromosome, QIGAPruningOptimizer
from .qisa import QISA, QuantumState, QISAPruningOptimizer

__all__ = [
    'QIGA',
    'QuantumChromosome', 
    'QIGAPruningOptimizer',
    'QISA',
    'QuantumState',
    'QISAPruningOptimizer'
]
