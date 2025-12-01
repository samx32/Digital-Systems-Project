# cifar10_pruning.py

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker


def load_cifar10_test(batch_size=64):
    """Load in the test set of CIFAR-10"""
    pass



def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the test datset"""
    pass


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    pass

def prune_model(model, amount: float = 0.3):
    """Apply unstructure pruning to all convolutional and linear layers in the model"""
    pass



def main():
    """Main function to load model, prune it, evaluate and measure energy consumption"""
    pass



if __name__ == "__main__":
    main()