# cifar10_structured_pruning.py - Structured (Channel) Pruning for CIFAR-10 CNN

"""
Structured Pruning Script

Unlike unstructured pruning (which zeros individual weights but keeps tensor
shapes identical), structured pruning removes entire output channels/filters
from Conv2d layers. This produces a genuinely smaller and faster model because
the tensor dimensions are physically reduced.

Approach:
  1. Rank each Conv2d layer's output channels by L1-norm of their weights
  2. Remove the lowest-norm channels (and corresponding input channels in the
     next layer, plus BatchNorm parameters)
  3. Rebuild the model with the reduced architecture
  4. Fine-tune to recover accuracy
  5. Save and evaluate

Usage:
    python -m src.classical_optimisation.cifar10_structured_pruning
"""

import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline.cifar10_cnn import CIFAR10CNN
from src.utils.energy_measurements import EnergyTracker


# ---------------------------------------------------------------------------
# Structured pruned model class (module-level so torch.save can pickle it)
# ---------------------------------------------------------------------------

class StructuredPrunedCNN(nn.Module):
    """A CIFAR-10 CNN with reduced channel counts from structured pruning."""
    def __init__(self, conv_layers, fc_layers):
        super().__init__()
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRUNING_LEVELS = [0.2, 0.4, 0.6]
FINE_TUNE_EPOCHS = 10           # More epochs — structured pruning is harsher
FINE_TUNE_LR = 0.001


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_data(batch_size=64):
    """Load CIFAR-10 train and test sets."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


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
    return total_loss / len(dataloader), correct / total


def count_parameters(model):
    """Count total and non-zero parameters."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
    return total, nonzero


def get_model_size_mb(model):
    """Get model size in MB."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Structured pruning helpers
# ---------------------------------------------------------------------------

def _get_conv_bn_pairs(model):
    """
    Walk the CIFAR10CNN Sequential to extract (Conv2d, BatchNorm2d) pairs
    and the first Linear layer (whose in_features depends on the last conv).

    Returns:
        conv_bn_pairs: list of (conv_layer, bn_layer) pairs
        first_linear: the nn.Linear that follows the conv blocks
    """
    conv_bn_pairs = []
    conv_layers = list(model.conv_layers)

    i = 0
    while i < len(conv_layers):
        layer = conv_layers[i]
        if isinstance(layer, nn.Conv2d):
            # Next layer should be BatchNorm2d
            bn = conv_layers[i + 1] if (i + 1 < len(conv_layers) and
                                         isinstance(conv_layers[i + 1], nn.BatchNorm2d)) else None
            conv_bn_pairs.append((layer, bn))
        i += 1

    # First layer in fc_layers after Flatten is the Linear
    first_linear = None
    for layer in model.fc_layers:
        if isinstance(layer, nn.Linear):
            first_linear = layer
            break

    return conv_bn_pairs, first_linear


def compute_channel_importance(conv: nn.Conv2d) -> torch.Tensor:
    """Rank output channels by L1-norm of their weight filters."""
    # conv.weight shape: (out_channels, in_channels, kH, kW)
    importance = conv.weight.data.abs().sum(dim=(1, 2, 3))  # (out_channels,)
    return importance


def structured_prune_model(model: CIFAR10CNN, amount: float) -> nn.Module:
    """
    Create a new, physically smaller model by removing low-importance output
    channels from each Conv2d layer.

    Args:
        model: Trained CIFAR10CNN
        amount: Fraction of channels to remove per layer (0–1)

    Returns:
        A new nn.Module with reduced channel counts and copied surviving weights.
    """
    model.eval()
    conv_bn_pairs, first_linear = _get_conv_bn_pairs(model)

    # Decide which channels to KEEP for each conv layer
    keep_indices = []
    for conv, _ in conv_bn_pairs:
        n_out = conv.out_channels
        n_keep = max(1, int(n_out * (1 - amount)))  # keep at least 1
        importance = compute_channel_importance(conv)
        _, sorted_idx = importance.sort(descending=True)
        keep = sorted_idx[:n_keep].sort().values  # keep in original order
        keep_indices.append(keep)

    # Build new Sequential for conv_layers
    new_conv_layers = []
    old_layers = list(model.conv_layers)

    conv_idx = 0  # tracks which conv/bn pair we're on
    prev_keep = None  # channel indices kept from previous conv (needed for input pruning)

    i = 0
    while i < len(old_layers):
        layer = old_layers[i]

        if isinstance(layer, nn.Conv2d):
            keep_out = keep_indices[conv_idx]
            old_conv = layer
            out_ch = len(keep_out)

            # Determine input channels to keep
            if prev_keep is not None:
                in_ch = len(prev_keep)
                # Subset weights: keep_out rows, prev_keep cols
                new_weight = old_conv.weight.data[keep_out][:, prev_keep].clone()
            else:
                in_ch = old_conv.in_channels  # first conv keeps all 3 RGB channels
                new_weight = old_conv.weight.data[keep_out].clone()

            new_conv = nn.Conv2d(in_ch, out_ch, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding,
                                 bias=old_conv.bias is not None)
            new_conv.weight.data = new_weight
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data[keep_out].clone()

            new_conv_layers.append(new_conv)

            # Handle BatchNorm if it follows
            if (i + 1 < len(old_layers) and isinstance(old_layers[i + 1], nn.BatchNorm2d)):
                old_bn = old_layers[i + 1]
                new_bn = nn.BatchNorm2d(out_ch)
                new_bn.weight.data = old_bn.weight.data[keep_out].clone()
                new_bn.bias.data = old_bn.bias.data[keep_out].clone()
                new_bn.running_mean = old_bn.running_mean[keep_out].clone()
                new_bn.running_var = old_bn.running_var[keep_out].clone()
                new_conv_layers.append(new_bn)
                i += 1  # skip the BN layer

            prev_keep = keep_out
            conv_idx += 1

        elif isinstance(layer, (nn.ReLU, nn.MaxPool2d, nn.Dropout)):
            new_conv_layers.append(copy.deepcopy(layer))

        i += 1

    # Build new fc_layers — the first Linear's in_features changes
    last_conv_channels = len(keep_indices[-1])
    # After 3 MaxPool2d (32→16→8→4), spatial dim = 4×4
    new_in_features = last_conv_channels * 4 * 4

    new_fc_layers = []
    first_linear_done = False
    for layer in model.fc_layers:
        if isinstance(layer, nn.Linear) and not first_linear_done:
            new_linear = nn.Linear(new_in_features, layer.out_features)
            # Copy the weights for surviving input features
            # The original Linear expects 128*4*4 = 2048 inputs, ordered as
            # (channel_0_pixel_0, channel_0_pixel_1, …, channel_127_pixel_15)
            # We need to pick the rows corresponding to the kept channels×16 pixels
            old_weight = layer.weight.data   # (512, 2048)
            spatial_size = 16  # 4×4 = 16
            col_indices = []
            for ch_idx in keep_indices[-1]:
                start = ch_idx.item() * spatial_size
                col_indices.extend(range(start, start + spatial_size))
            col_indices = torch.tensor(col_indices, dtype=torch.long)
            new_linear.weight.data = old_weight[:, col_indices].clone()
            new_linear.bias.data = layer.bias.data.clone()
            new_fc_layers.append(new_linear)
            first_linear_done = True
        else:
            new_fc_layers.append(copy.deepcopy(layer))

    # Assemble into the module-level StructuredPrunedCNN class
    return StructuredPrunedCNN(
        nn.Sequential(*new_conv_layers),
        nn.Sequential(*new_fc_layers),
    )


def fine_tune(model, train_loader, criterion, optimizer, device, epochs=5):
    """Fine-tune the pruned model."""
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
        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"    Fine-tune [{epoch}/{epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run structured pruning experiments at multiple levels."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = load_cifar10_data()
    criterion = nn.CrossEntropyLoss()
    save_dir = './data/models'
    os.makedirs(save_dir, exist_ok=True)

    # --- Baseline ---
    print("\n" + "=" * 65)
    print("BASELINE MODEL")
    print("=" * 65)
    baseline = CIFAR10CNN().to(device)
    baseline.load_state_dict(
        torch.load('./data/models/cifar10_cnn.pth', map_location=device, weights_only=True)
    )
    total_params, _ = count_parameters(baseline)
    baseline_size = get_model_size_mb(baseline)
    baseline_loss, baseline_acc = evaluate_model(baseline, test_loader, criterion, device)
    print(f"Accuracy: {baseline_acc * 100:.2f}%  |  Params: {total_params:,}  |  Size: {baseline_size:.2f} MB")

    results = []

    for amount in PRUNING_LEVELS:
        tag = int(amount * 100)
        print(f"\n{'=' * 65}")
        print(f"STRUCTURED PRUNING — {tag}% CHANNELS REMOVED")
        print(f"{'=' * 65}")

        # Reload and structurally prune
        model = CIFAR10CNN()
        model.load_state_dict(
            torch.load('./data/models/cifar10_cnn.pth', map_location='cpu', weights_only=True)
        )
        pruned_model = structured_prune_model(model, amount=amount)
        pruned_model.to(device)

        p_total, p_nonzero = count_parameters(pruned_model)
        p_size = get_model_size_mb(pruned_model)
        print(f"  Parameters: {p_total:,}  (reduction: {(1 - p_total / total_params) * 100:.1f}%)")
        print(f"  Model size: {p_size:.2f} MB  (reduction: {(1 - p_size / baseline_size) * 100:.1f}%)")

        # Evaluate BEFORE fine-tuning
        pre_loss, pre_acc = evaluate_model(pruned_model, test_loader, criterion, device)
        print(f"  Accuracy before fine-tuning: {pre_acc * 100:.2f}%")

        # Fine-tune
        print(f"  Fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
        opt = torch.optim.Adam(pruned_model.parameters(), lr=FINE_TUNE_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINE_TUNE_EPOCHS)
        pruned_model.train()
        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                opt.zero_grad()
                outputs = pruned_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            scheduler.step()
            acc = correct / total
            avg = total_loss / len(train_loader)
            print(f"    Fine-tune [{epoch}/{FINE_TUNE_EPOCHS}] Loss: {avg:.4f}, Acc: {acc:.4f}")

        # Evaluate AFTER fine-tuning with energy tracking
        with EnergyTracker(experiment_name=f"cifar10_struct_pruned_{tag}") as tracker:
            post_loss, post_acc = evaluate_model(pruned_model, test_loader, criterion, device)
            tracker.set_accuracy(post_acc)

        print(f"  Accuracy after fine-tuning: {post_acc * 100:.2f}%")
        print(f"  Accuracy drop from baseline: {(baseline_acc - post_acc) * 100:.2f}%")
        print(f"  Accuracy recovered by fine-tuning: {(post_acc - pre_acc) * 100:.2f}%")
        print(f"  Energy metrics: {tracker.metrics}")

        # Save (full model — architecture varies per pruning level)
        save_path = os.path.join(save_dir, f'cifar10_struct_pruned_{tag}.pth')
        torch.save(pruned_model, save_path)
        print(f"  Saved to {save_path}")

        results.append({
            'method': f'Structured {tag}%',
            'accuracy': post_acc,
            'params': p_total,
            'size': p_size,
        })

    # --- Summary ---
    print(f"\n{'=' * 65}")
    print("STRUCTURED PRUNING SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Params':>12} {'Size (MB)':>10} {'Param Red.':>11}")
    print("-" * 70)
    print(f"{'Baseline':<25} {baseline_acc * 100:>9.2f}% {total_params:>12,} {baseline_size:>10.2f} {'—':>11}")
    for r in results:
        red = (1 - r['params'] / total_params) * 100
        print(f"{r['method']:<25} {r['accuracy'] * 100:>9.2f}% {r['params']:>12,} {r['size']:>10.2f} {red:>10.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
