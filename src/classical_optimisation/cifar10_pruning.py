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
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_dataset = datasets.CIFAR10(
        root = "./data",
        train = False, 
        download = True, 
        transform = test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader



def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the test datset"""
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

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    total = 0 
    nonzero = 0 
    for param in model.parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()
    return total, nonzero


def prune_model(model, amount: float = 0.3):
    """Apply unstructure pruning to all convolutional and linear layers in the model"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    return model        



def main():
    """Main function to load model, prune it, evaluate and measure energy consumption"""
    # Device config



if __name__ == "__main__":
    main()