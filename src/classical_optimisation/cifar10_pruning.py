# cifar10_pruning.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_data(batch_size=64):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



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


def fine_tune(model, train_loader, criterion, optimizer, device, epochs=5):
    """Fine-tune the pruned model to recover accuracy."""
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
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Fine-tune Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    
    return model        



def main():
    """Main function to load model, prune it, fine-tune, evaluate and measure energy consumption"""
    set_seed(42)
    
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load train and test data
    train_loader, test_loader = load_cifar10_data()

    # Load model
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True))
    criterion = nn.CrossEntropyLoss()

    # Evaluate original model
    print("\n=== BASELINE MODEL ===")
    total_params, nonzero_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {nonzero_params:,}")

    with EnergyTracker(experiment_name="cifar10_baseline") as tracker:
        baseline_loss, baseline_acc = evaluate_model(model, test_loader, criterion, device)

    print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Baseline energy consumption: {tracker.metrics}")

    # Test pruning levels with fine-tuning
    pruning_levels = [0.2, 0.4, 0.6, 0.8]
    fine_tune_epochs = 5

    for amount in pruning_levels:
        print(f"\n=== PRUNING AMOUNT: {amount * 100:.0f}% ===")
        
        # Reload fresh model and apply pruning
        model = CIFAR10CNN().to(device)
        model.load_state_dict(torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True))
        model = prune_model(model, amount=amount)

        # Count params after pruning
        total_params, nonzero_params = count_parameters(model)
        sparsity = 1 - (nonzero_params / total_params)
        print(f"Non-zero parameters: {nonzero_params:,} ({sparsity * 100:.1f}% sparse)")

        # Evaluate BEFORE fine-tuning
        pre_ft_loss, pre_ft_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Accuracy before fine-tuning: {pre_ft_acc * 100:.2f}%")

        # Fine-tune the pruned model
        print(f"Fine-tuning for {fine_tune_epochs} epochs...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        model = fine_tune(model, train_loader, criterion, optimizer, device, epochs=fine_tune_epochs)

        # Evaluate AFTER fine-tuning with energy tracking
        with EnergyTracker(experiment_name=f"cifar10_pruned_{int(amount * 100)}_finetuned") as tracker:
            pruned_loss, pruned_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Accuracy after fine-tuning: {pruned_acc * 100:.2f}%")
        print(f"Accuracy drop from baseline: {(baseline_acc - pruned_acc) * 100:.2f}%")
        print(f"Accuracy recovered by fine-tuning: {(pruned_acc - pre_ft_acc) * 100:.2f}%")
        print(f"Energy metrics: {tracker.metrics}")



if __name__ == "__main__":
    main()