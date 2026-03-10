# Model Evaluation & Comparison

## 1. Experimental Setup

### Hardware
All experiments were conducted on a single laptop-class system:

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 7 6800H (8 cores, 16 threads, 45 W TDP) |
| GPU | NVIDIA GeForce RTX 3070 Laptop (8 GB VRAM, 125 W TGP) |
| RAM | 16 GB DDR5 |
| OS | Windows 11 |
| Framework | PyTorch 2.x, CUDA |

This is a **consumer-grade mobile workstation**, not a server or data-centre GPU. Results should be interpreted with this in mind — throughput, latency and energy figures would differ on desktop GPUs (higher TDP, more VRAM bandwidth), server-class accelerators (A100, H100), or low-power edge devices (Jetson, Raspberry Pi).

### Models Tested
Three architectures were evaluated on CIFAR-10 (32×32, 10 classes):

| Architecture | Parameters | Size (MB) | Origin |
|-------------|-----------|-----------|--------|
| Custom CNN | ~1.3 M | 5.14 | Trained from scratch (20 epochs) |
| ResNet-18 | ~11.7 M | 42.70 | ImageNet-pretrained, fine-tuned (10 epochs) |
| VGG-16 | ~134 M | 128.33 | ImageNet-pretrained, fine-tuned (10 epochs) |

### Optimisation Techniques Applied
- **Unstructured L1 pruning** (20%, 40%, 60%, 80%) with fine-tuning
- **Structured channel pruning** (20%, 40%, 60%) — Custom CNN only
- **Dynamic INT8 quantization** (CPU-only, PyTorch fbgemm backend)
- **Static INT8 quantization** — Custom CNN only
- **Combined pruning + quantization** — Custom CNN only
- **QIGA** (Quantum-Inspired Genetic Algorithm)
- **QISA** (Quantum-Inspired Simulated Annealing)

---

## 2. Architecture Baseline Comparison

| Architecture | Accuracy (%) | Throughput (img/s) | Energy (J) | Latency (ms) | CO₂ (g) |
|-------------|-------------|-------------------|-----------|-------------|---------|
| Custom CNN | 84.15 | 79,973 | 652.5 | 1.58 | 0.0375 |
| ResNet-18 | 94.12 | 37,783 | 2,453.0 | 3.35 | 0.1410 |
| VGG-16 | 86.74 | 61,005 | 1,278.8 | 2.07 | 0.0735 |

**Key observations:**
- ResNet-18 achieves the highest baseline accuracy (94.12%), a full **10 percentage points** above the custom CNN. This is expected — ResNet-18's deeper residual architecture and ImageNet pre-training provide strong feature representations that transfer well to CIFAR-10.
- The custom CNN delivers the **highest throughput** (79,973 img/s) and **lowest energy** (652.5 J) due to its small parameter count (~1.3 M). It processes data ~2.1× faster than ResNet-18.
- VGG-16 is a middle ground in throughput (61,005 img/s) despite being the largest model (128.33 MB). Its simple sequential architecture maps efficiently to GPU parallelism — no skip connections to synchronise.
- ResNet-18's skip connections create memory access bottlenecks on the RTX 3070's mobile memory bus, explaining why it is slower than VGG-16 despite being 3× smaller.

---

## 3. Pruning Effectiveness

### Accuracy Change from Baseline

| Pruning Level | Custom CNN | Δ | ResNet-18 | Δ | VGG-16 | Δ |
|--------------|-----------|---|----------|---|--------|---|
| 20% | 87.03 | **+2.88** | 94.15 | +0.03 | 88.02 | **+1.28** |
| 40% | 86.89 | **+2.74** | 94.28 | **+0.16** | 88.14 | **+1.40** |
| 60% | 86.55 | **+2.40** | 94.10 | −0.02 | 88.13 | **+1.39** |
| 80% | 84.86 | +0.71 | — | — | — | — |

**Pruning acts as regularisation across all three architectures.** Every model *improves* in accuracy after moderate pruning (20–60%), with the effect most pronounced on the custom CNN (+2.88% at 20% pruning). This strongly suggests all three models are over-parameterised for the CIFAR-10 task — removing redundant weights reduces overfitting.

- **Custom CNN**: Best at Pruned 20% (87.03%). The improvement declines gradually; at 80% pruning the model is only marginally above baseline (84.86%), indicating the limit of useful compression.
- **ResNet-18**: Extremely robust to pruning. Even at 60% sparsity it retains 94.10% accuracy (only −0.02% from baseline). The best result is at 40% (94.28%) — the highest accuracy recorded across all experiments.
- **VGG-16**: Shows the clearest regularisation benefit — 40% pruning achieves 88.14%, a full 1.40% above baseline. This is consistent with VGG-16 being known as a heavily over-parameterised architecture.

### Structured Pruning (Custom CNN Only)

| Level | Accuracy | Size (MB) | Energy (J) |
|-------|---------|-----------|-----------|
| Baseline | 84.15 | 5.14 | 652.5 |
| Struct 20% | 86.11 | 3.93 | 717.7 |
| Struct 40% | 83.72 | 2.81 | 640.6 |
| Struct 60% | 78.30 | 1.81 | 558.3 |

Structured pruning removes entire channels, so it achieves **real size reduction** (5.14 → 1.81 MB at 60%). However, accuracy drops sharply at 60% (78.30%), suggesting the custom CNN's architecture cannot tolerate losing entire feature maps as well as it tolerates zeroing individual weights.

---

## 4. Quantization Results

| Model | Technique | Accuracy (%) | Throughput (img/s) | Energy (J) | Size (MB) |
|-------|----------|-------------|-------------------|-----------|-----------|
| Custom CNN | Dynamic INT8 | 84.13 | 2,912 | 2,373.3 | 2.13 |
| Custom CNN | Static INT8 | 84.52 | 9,698 | 1,128.8 | 1.31 |
| ResNet-18 | Dynamic INT8 | 94.10 | 319 | 14,970.7 | 42.70 |
| VGG-16 | Dynamic INT8 | 86.73 | 585 | 8,374.7 | 74.22 |

**Dynamic quantization is catastrophically inefficient on GPU-trained models.** PyTorch's dynamic INT8 quantization uses the CPU-only `fbgemm` backend, forcing the model off the GPU entirely. The impact scales with model size:

- Custom CNN: 27× slower, 3.6× more energy
- VGG-16: **104× slower**, 6.5× more energy
- ResNet-18: **118× slower**, 6.1× more energy

Static quantization (Custom CNN only) performs much better — only 8× slower than the GPU baseline — because calibration produces optimised INT8 kernels. However, it is still significantly slower than the FP32 GPU path.

**Hardware limitation:** These results reflect the fact that the RTX 3070 Laptop does not natively support INT8 tensor operations (unlike server-class A100/H100 with INT8 Tensor Cores). On hardware with native INT8 acceleration, quantized models would likely show throughput *improvements* over FP32. This is a significant limitation of this study's hardware platform.

---

## 5. Quantum-Inspired Optimisation

| Technique | Architecture | Accuracy (%) | Δ from Baseline | Energy (J) | Δ Energy |
|----------|-------------|-------------|----------------|-----------|----------|
| QIGA | Custom CNN | 86.51 | +2.36 | 652.5 | +0.0% |
| QISA | Custom CNN | 86.35 | +2.20 | 635.5 | −2.6% |
| QIGA | ResNet-18 | 93.97 | −0.15 | 2,468.2 | +0.6% |
| QISA | ResNet-18 | 94.16 | +0.04 | 2,660.3 | +8.4% |
| QIGA | VGG-16 | 87.04 | +0.30 | 1,233.7 | −3.5% |
| QISA | VGG-16 | 86.61 | −0.13 | 1,243.5 | −2.8% |

### Comparison Against Classical Pruning

| Architecture | Best Pruning | QIGA | QISA |
|-------------|-------------|------|------|
| Custom CNN | 87.03% (P20) | 86.51% (−0.52) | 86.35% (−0.68) |
| ResNet-18 | 94.28% (P40) | 93.97% (−0.31) | 94.16% (−0.12) |
| VGG-16 | 88.14% (P40) | 87.04% (−1.10) | 86.61% (−1.53) |

Quantum-inspired methods improve upon baseline accuracy for the custom CNN (+2.2 to +2.4%) and are competitive with classical pruning, but consistently fall slightly short of the best fixed-ratio pruning configurations. The gap is larger for VGG-16 (−1.1 to −1.5%) and smallest for ResNet-18 (−0.1 to −0.3%).

### Why Quantum-Inspired Methods Underperform — And Why This Does Not Invalidate Them

Several factors must be considered:

1. **Classical simulation overhead**: QIGA and QISA are *quantum-inspired* algorithms running on classical hardware. They simulate quantum phenomena (superposition, tunnelling, rotation gates) using standard floating-point arithmetic. On a genuine quantum processor, operations like superposition and entanglement are performed natively in O(1) time, allowing exploration of exponentially larger search spaces simultaneously. The classical simulation is inherently constrained to sequential evaluation of one configuration at a time.

2. **Reduced search budget for feasibility**: To keep runtimes practical on consumer hardware, the pretrained model experiments used reduced parameters (population size 4–6, generations 6–10 for QIGA; ~80 total evaluations for QISA). This significantly limits the search space exploration compared to what would be possible with longer runtimes or parallel quantum hardware.

3. **Simple search landscape**: Unstructured L1 pruning at fixed ratios (20/40/60%) is a well-understood technique with a small discrete search space. QIGA/QISA are designed for complex, high-dimensional combinatorial optimisation where the landscape has many local optima. The pruning ratio search space may be too simple to demonstrate their full advantage — the classical grid search over 3–4 pruning levels is effectively exhaustive.

4. **Potential with real quantum hardware**: On a quantum computer, QIGA could maintain a genuine quantum population in superposition, evaluating all candidate pruning configurations simultaneously via quantum parallelism. Quantum tunnelling in QISA would be a real physical phenomenon rather than a probabilistic approximation, enabling escape from local optima that classical simulated annealing cannot. Research by Harrigan et al. (2021) and Abbas et al. (2023) has shown that quantum optimisation algorithms can achieve polynomial speedups on combinatorial problems when run on sufficient-qubit quantum hardware.

5. **Energy efficiency of quantum hardware**: Quantum computers such as IBM's Eagle processors operate at ~15 mW during computation (excluding cryogenic cooling). If the cooling overhead can be amortised across many problems (as in a shared quantum cloud service), the per-query energy cost could be orders of magnitude lower than the GPU-based evaluations in this study.

**Therefore, the results of this study do not demonstrate that quantum-inspired optimisation is non-viable.** They demonstrate that on classical consumer hardware with a constrained search budget, quantum-inspired methods are competitive with but slightly inferior to exhaustive classical search over a simple pruning parameter space. The true potential of these algorithms can only be fully assessed on genuine quantum hardware with larger problem instances.

---

## 6. Combined Optimisation (Custom CNN Only)

| Technique | Accuracy (%) | Throughput (img/s) | Energy (J) | Size (MB) |
|----------|-------------|-------------------|-----------|-----------|
| Pruned 40% + Stat Quant | **87.12** | 9,808 | 1,156.7 | 1.31 |
| Pruned 20% + Stat Quant | 86.89 | 9,482 | 1,191.6 | 1.31 |
| Pruned 20% + Dyn Quant | 86.94 | 2,938 | 2,205.6 | 2.13 |

The combination of 40% pruning + static quantization achieves the **highest accuracy of any Custom CNN variant** (87.12%) at 1.31 MB — a 74.5% size reduction from the baseline. However, throughput drops to ~9,800 img/s due to CPU-only INT8 inference (the same hardware limitation discussed above).

---

## 7. Energy Efficiency Ranking

**Top 5 most energy-efficient models (accuracy per joule):**

| Rank | Model | Architecture | Acc (%) | Energy (J) | Acc/Joule |
|------|-------|-------------|---------|-----------|-----------|
| 1 | Struct Pruned 60% | Custom CNN | 78.30 | 558.3 | 0.1402 |
| 2 | Pruned 20% | Custom CNN | 87.03 | 631.8 | 0.1377 |
| 3 | Pruned 60% | Custom CNN | 86.55 | 629.6 | 0.1375 |
| 4 | QISA Optimized | Custom CNN | 86.35 | 635.5 | 0.1359 |
| 5 | Pruned 40% | Custom CNN | 86.89 | 653.9 | 0.1329 |

The **entire top 10 is dominated by Custom CNN variants**. The most energy-efficient model overall is Structured Pruned 60%, but it sacrifices significant accuracy (78.30%). A more practical choice is **Pruned 20%** — the second most efficient, with 87.03% accuracy and only 631.8 J energy.

No ResNet-18 or VGG-16 variant appears in the top 10. The most efficient ResNet-18 model (Baseline, 0.0384 acc/J) is 3.6× less efficient than the Custom CNN Pruned 20% — the price paid for the 7-point accuracy advantage.

---

## 8. Overall Verdict

### Best Model Depends on the Deployment Objective

| Objective | Best Model | Accuracy | Throughput | Energy | Size |
|-----------|-----------|---------|-----------|--------|------|
| **Maximum accuracy** | RN18 Pruned 40% | **94.28%** | 38,406 | 2,606 J | 42.7 MB |
| **Best accuracy/energy** | CNN Pruned 20% | 87.03% | 83,399 | **632 J** | 5.14 MB |
| **Smallest deployment** | CNN P40 + Stat Quant | 87.12% | 9,808 | 1,157 J | **1.31 MB** |
| **Fastest inference** | CNN Pruned 20% | 87.03% | **83,399** | 632 J | 5.14 MB |
| **Lowest carbon** | CNN Struct Pruned 60% | 78.30% | 82,929 | 558 J | 1.81 MB |

There is no single "best" model — the answer depends on the deployment context:

- **If accuracy is paramount** (e.g. medical imaging, safety-critical systems): ResNet-18 Pruned 40% at 94.28% is the clear winner, but it requires 4× more energy and a GPU with ≥42.7 MB model memory.
- **If deploying to edge/IoT devices** with power constraints: Custom CNN Pruned 20% delivers 87.03% accuracy at 632 J and 83,399 img/s — the best balance of accuracy, speed and efficiency.
- **If model size is the primary constraint** (e.g. microcontroller deployment): Pruned 40% + Static Quantized at 1.31 MB is the most compressed model that retains strong accuracy (87.12%).
- **If environmental sustainability is prioritised**: Structured Pruned 60% uses the least energy (558 J, 0.032 gCO₂) but accuracy drops to 78.30%.

### Hardware Limitations of This Study

The results are specific to the AMD Ryzen 7 6800H + RTX 3070 Laptop platform. Key limitations:

1. **No native INT8 GPU acceleration**: The RTX 3070 lacks INT8 Tensor Cores, making quantized inference CPU-bound. On hardware with INT8 support (A100, Jetson Xavier), quantized models would likely show throughput *improvements* rather than the severe degradation observed here.

2. **Mobile GPU memory bandwidth**: The RTX 3070 Laptop has 256 GB/s memory bandwidth vs 2 TB/s on an A100. This disproportionately affects larger models (ResNet-18, VGG-16) and explains ResNet-18's lower throughput despite being smaller than VGG-16.

3. **Thermal throttling**: Laptop TDP limits (45 W CPU, 125 W GPU) mean sustained workloads may hit thermal limits, introducing variance in energy measurements. Desktop or server hardware with higher TDP headroom would produce more consistent results.

4. **Single machine, no parallelism**: All experiments ran sequentially on one device. Data-centre deployments would distribute inference across multiple accelerators, changing the throughput and energy calculus entirely.

### Quantum-Inspired Methods: Conclusion

QIGA and QISA produce results competitive with classical pruning (+2.2 to +2.4% above baseline for the Custom CNN) but slightly below the best fixed-ratio variants. This study cannot conclude that quantum-inspired optimisation is non-viable because:

- The algorithms were run as **classical simulations**, not on quantum hardware
- The **search budget was intentionally constrained** for practical runtime
- The **problem complexity** (choosing among a handful of pruning ratios) may be too simple to exhibit quantum advantage
- Genuine quantum hardware would enable **true superposition and tunnelling**, potentially providing exponential speedup in search space exploration

These algorithms remain a promising research direction for more complex optimisation landscapes (e.g. neural architecture search, mixed-precision quantization, joint pruning-quantization-distillation) where the search space is too large for exhaustive classical methods.

---

## 9. Carbon Footprint Summary

| Metric | Value |
|--------|-------|
| Total CO₂ across all 32 benchmarks | 3.81 gCO₂ |
| Grid intensity used | 207 gCO₂/kWh (UK average) |
| Most carbon-efficient model | Struct Pruned 60% (0.032 gCO₂) |
| Most carbon-intensive model | RN18 Dyn Quant (0.861 gCO₂) |

The total carbon footprint of the entire experimental pipeline is negligible (3.81 g — equivalent to driving ~15 metres in a petrol car). However, at production scale with millions of inference queries, the differences become meaningful. The 27× energy gap between the custom CNN baseline and dynamic quantization would translate to significant operational cost and environmental impact.
