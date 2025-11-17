# iris_mlp.py

import torch
import torch.nn as nn

class IrisMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for classifying Iris dataset.

    Input: 4 features (sepal length, sepal width, petal length, petal width)
    Output: 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica)

    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 16, output_size: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)