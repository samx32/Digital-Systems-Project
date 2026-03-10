# cifar10_training_baseline.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_data(batch_size: int = 64):
    """Load and preprocess CIFAR-10 dataset."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root='data/cifar10', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='data/cifar10', train=False, download=True, transform=test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size=batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialise model, loss function, and optimiser
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training loop with energy tracking
    print("\nStarting training with energy tracking...")
    
    with EnergyTracker(experiment_name="cifar10_baseline") as tracker:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

            print(
                f'Epoch [{epoch}/{num_epochs}] '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}'
            )

    # Final test accuracy
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    tracker.set_accuracy(final_test_acc)
    print(f'\nFinal Test Accuracy on CIFAR-10: {final_test_acc * 100:.2f}%')

    # Save model
    os.makedirs('data/models', exist_ok=True)
    torch.save(model.state_dict(), 'data/models/cifar10_cnn.pth')
    print("Model saved to data/models/cifar10_cnn.pth")

    print("\nEnergy measurement for baseline training complete.")
    print(f"Energy metrics: {tracker.metrics}")

if __name__ == '__main__':
    main()