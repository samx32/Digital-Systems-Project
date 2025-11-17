# iris_training_baseline.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from iris_mlp import IrisMLP


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_iris_data(test_size: float = 0.2):
    """
    Load and preprocess the Iris dataset.
    Splits the data into training and testing sets.
    """
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Standardise features 
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=42, stratify=y
    )


    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader
 


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """ 
    Reusable function to train a chosen model for one epoch and return loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

        

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * x_batch.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    train_loader, test_loader = load_iris_data(test_size=0.2)

    # Initialise the model, loss function, and optimiser
    model = IrisMLP(input_size=4, hidden_size=16, output_size=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f'Epoch [{epoch}/{num_epochs}] '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}'
            )

    # Final test accuracy
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print('\nFinal Test Accuracy on Iris: {:.2f}%'.format(final_test_acc * 100))


if __name__ == '__main__':
    main()

