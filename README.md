# Digital Systems Project

**Investigating Classical and Quantum-Inspired Optimisation Approaches for Environmentally Sustainable Artificial Intelligence**

Final year university project exploring how pruning, quantization, and quantum-inspired algorithms can reduce the energy footprint of neural network inference on CIFAR-10.

---

## Project Structure

```
Digital-Systems-Project/
├── src/
│   ├── main.py                          # Full pipeline runner (all 11 stages)
│   ├── baseline/
│   │   ├── cifar10_cnn.py               # Custom CNN architecture
│   │   ├── cifar10_training_baseline.py # Train & evaluate custom CNN
│   │   ├── iris_mlp.py                  # Iris MLP architecture
│   │   ├── iris_training_baseline.py    # Train & evaluate Iris MLP
│   │   ├── resnet18_cifar10.py          # ResNet-18 fine-tuned on CIFAR-10
│   │   └── vgg16_cifar10.py             # VGG-16 fine-tuned on CIFAR-10
│   ├── classical_optimisation/
│   │   ├── cifar10_pruning.py           # Unstructured pruning (20/40/60/80%)
│   │   ├── cifar10_structured_pruning.py# Structured (filter) pruning (20/40/60%)
│   │   ├── cifar10_quantization.py      # Dynamic & static quantization
│   │   ├── cifar10_combined_optimization.py # Pruning + quantization pipeline
│   │   └── pretrained_optimization.py  # Classical optimisation on ResNet-18/VGG-16
│   ├── quantum_inspired/
│   │   ├── qiga.py                      # Quantum-Inspired Genetic Algorithm (QIGA)
│   │   ├── qisa.py                      # Quantum-Inspired Simulated Annealing (QISA)
│   │   ├── cifar10_quantum_optimization.py  # QIGA/QISA on custom CNN
│   │   └── pretrained_quantum_optimization.py # QIGA/QISA on ResNet-18/VGG-16
│   ├── evaluation/
│   │   ├── inference_benchmark.py       # Side-by-side latency/throughput benchmark
│   │   ├── generate_analysis.py         # Summary statistics and comparison tables
│   │   ├── generate_graphs.py           # All result graphs and animations
│   │   └── hardware_projections.py      # Server-grade & quantum hardware projections
│   └── utils/
│       └── energy_measurements.py       # CPU/GPU energy tracking via py3nvml + psutil
├── data/
│   ├── models/                          # Saved model checkpoints (.pth)
│   └── results/                         # Energy metrics CSVs and benchmark output
├── analysis/
│   ├── MODEL_EVALUATION.md              # Written analysis and findings
│   ├── graphs/                          # Generated plots and animations
│   ├── tables/                          # Result CSVs (accuracy, energy, size, etc.)
│   └── projections/                     # Hardware projection graphs and tables
├── TEST_PLAN.md                         # Test plan
└── requirements.txt
```

---

## Setup

**Python 3.11+ recommended.**

```bash
pip install -r requirements.txt
```

> GPU support requires a CUDA-capable NVIDIA GPU. CPU-only runs are supported but significantly slower for training.

---

## Running the Experiments

### Full pipeline (all 11 stages in order)

```bash
python -m src.main
```

Stages run in sequence:
1. Train baseline CIFAR-10 CNN
2. Unstructured pruning (20 / 40 / 60 / 80%)
3. Structured pruning (20 / 40 / 60%)
4. Dynamic & static quantization
5. Combined pruning + quantization
6. QIGA & QISA on custom CNN
7. Train ResNet-18 baseline (pretrained, fine-tuned)
8. Train VGG-16 baseline (pretrained, fine-tuned)
9. Classical optimisation on ResNet-18 / VGG-16
10. Quantum-inspired optimisation on ResNet-18 / VGG-16
11. Inference benchmark (all models)

### Individual stages

```bash
# Baselines
python -m src.baseline.iris_training_baseline
python -m src.baseline.cifar10_training_baseline
python -m src.baseline.resnet18_cifar10
python -m src.baseline.vgg16_cifar10

# Classical optimisation
python -m src.classical_optimisation.cifar10_pruning
python -m src.classical_optimisation.cifar10_structured_pruning
python -m src.classical_optimisation.cifar10_quantization
python -m src.classical_optimisation.cifar10_combined_optimization
python -m src.classical_optimisation.pretrained_optimization

# Quantum-inspired optimisation
python -m src.quantum_inspired.cifar10_quantum_optimization
python -m src.quantum_inspired.pretrained_quantum_optimization

# Evaluation
python -m src.evaluation.inference_benchmark
python -m src.evaluation.generate_analysis
python -m src.evaluation.generate_graphs
python -m src.evaluation.hardware_projections
```

---

## Outputs

| Location | Contents |
|---|---|
| `data/models/` | Trained model checkpoints (`.pth`) for all configurations |
| `data/results/` | Per-run energy metrics CSVs and inference benchmark results |
| `analysis/tables/` | Summary CSVs: accuracy, energy, model size, throughput |
| `analysis/graphs/` | Bar charts, scatter plots, radar chart, animations (`.gif`) |
| `analysis/projections/` | Hardware projection graphs and tables |
| `analysis/MODEL_EVALUATION.md` | Full written analysis and discussion |

---

## Optimisation Techniques

| Category | Technique | Models |
|---|---|---|
| Classical | Unstructured pruning | Custom CNN |
| Classical | Structured (filter) pruning | Custom CNN |
| Classical | Dynamic quantization | Custom CNN, ResNet-18, VGG-16 |
| Classical | Static quantization | Custom CNN |
| Classical | Pruning + quantization combined | Custom CNN |
| Classical | Pretrained model optimisation | ResNet-18, VGG-16 |
| Quantum-inspired | QIGA (Quantum-Inspired Genetic Algorithm) | Custom CNN, ResNet-18, VGG-16 |
| Quantum-inspired | QISA (Quantum-Inspired Simulated Annealing) | Custom CNN, ResNet-18, VGG-16 |
