# Hardware Projection Analysis

> Generated: 2026-03-05 21:07
> **All Tier 2 and Tier 3 figures are projections/estimates, not measurements.**
> Projection methodology is documented in `src/evaluation/hardware_projections.py`.

## Overview

This analysis projects the benchmark results (measured on an RTX 3070 Laptop + Ryzen 7 6800H)
onto server-grade and theoretical quantum hardware to contextualise how the optimisation
techniques would perform in a production or research environment.

---

## Tier 1 — Real Hardware (RTX 3070 Laptop + Ryzen 7 6800H)

| Metric | Value |
|---|---|
| GPU | NVIDIA RTX 3070 Laptop (15 TFLOPS FP32, 80W TGP) |
| CPU | AMD Ryzen 7 6800H (45W TDP, ~384 GFLOPS) |
| Best GPU throughput | 83,399 img/s (Pruned 20%) |
| Dynamic quantization | Runs on CPU only — 27–118× slower than GPU baseline |

---

## Tier 2 — Projected Server Hardware

### NVIDIA A100 80GB SXM4

| Spec | Value | Source |
|---|---|---|
| FP32 TFLOPS | 312 | NVIDIA A100 datasheet |
| INT8 TOPS | 624 | NVIDIA A100 datasheet |
| TDP | 400W | NVIDIA A100 datasheet |
| FP32 speedup vs RTX 3070 Laptop | ~20.8× raw TFLOPS | Sub-linear: ~9.7× real throughput |

**Key finding:** On an A100, dynamic quantization becomes far more competitive.
The INT8 TOPS ratio vs the Ryzen 7 6800H is 624 / 0.768 ≈ 812×,
meaning quantized models that underperform on the laptop CPU would be
dramatically faster on server-grade hardware with INT8 tensor core support.

### AMD EPYC 7763 (64-core server CPU)

| Spec | Value | Source |
|---|---|---|
| Cores | 64 (128 threads) | AMD EPYC 7763 datasheet |
| TDP | 280W | AMD EPYC 7763 datasheet |
| Multi-thread speedup vs 6800H | ~7.2× | Cinebench R23 multi-thread scaling |

**Key finding:** Quantized models running on an EPYC 7763 would be ~5–7× faster
than on the Ryzen 7 6800H, making dynamic quantization a viable deployment strategy
for CPU inference servers.

---

## Quantum-Inspired Methods on Server Hardware

QIGA and QISA optimised models run standard FP32 GPU inference — the quantum-inspired
component only affects the weight-search phase (optimisation), not inference.
They therefore benefit from GPU TFLOPS scaling exactly like any other FP32 model.

| Model | Architecture | RTX 3070 Img/s | T4 Img/s *(proj)* | A100 Img/s *(proj)* | RTX 3070 Energy (J) | A100 Energy (J) |
|---|---|---:|---:|---:|---:|---:|
| QIGA Optimized | Custom CNN — QIGA | 82,702 | 248,391 | 805,501 | 652 | 4 |
| QISA Optimized | Custom CNN — QISA | 82,064 | 246,472 | 799,279 | 636 | 4 |
| RN18 QIGA | ResNet-18 — QIGA | 37,375 | 112,252 | 364,019 | 2,468 | 9 |
| RN18 QISA | ResNet-18 — QISA | 37,852 | 113,687 | 368,673 | 2,660 | 9 |
| VGG16 QIGA | VGG-16 — QIGA | 60,912 | 182,945 | 593,270 | 1,234 | 6 |
| VGG16 QISA | VGG-16 — QISA | 61,736 | 185,420 | 601,294 | 1,243 | 6 |

> Throughput is for 10,000-image evaluation. Energy is Joules for the same task.
> T4 and A100 columns are **projected** (TFLOPS ratio^0.75 scaling, not measured).

**Key finding:** QIGA/QISA models achieve the same throughput and energy scaling as
the FP32 baseline on server GPUs. The quantum-inspired optimisation delivers accuracy
improvements with no inference-time compute overhead — making them the most favourable
option for server deployment at high accuracy.

---

## Tier 3 — Theoretical Quantum Hardware

### Quantum Speedup Modelling

| Source | Value | Reference |
|---|---|---|
| QIGA superposition speedup | ~8× (= population size) | Brassard et al. 2002; Farhi et al. 2014 (QAOA) |
| QISA tunnelling speedup | ~4× | Finnila et al. 1994 (Quantum Annealing) |
| Near-term NISQ cap | ≤100× practical | Preskill 2018 |
| IBM Eagle qubits | 127 | IBM Quantum (2023) |
| Google Sycamore qubits | 53 | Arute et al. Nature 2019 |
| Cryogenic power | ~25 kW | Krinner et al. Rev. Mod. Phys. 2019 |

**Important caveat:** The cryogenic cooling system required to maintain quantum coherence
consumes ~25 kW continuously — orders of magnitude more than the GPU system.
However, the **compute time** reduction is significant, and as quantum hardware matures,
room-temperature quantum computing (e.g. photonic quantum systems) could eliminate this overhead.

**QIGA on quantum hardware:** The population of chromosomes can be evaluated simultaneously
via quantum superposition rather than sequentially. With a population of 8, this gives
an 8× reduction in compute time — consistent with Grover's algorithm bounds for near-term devices.

**QISA on quantum hardware:** Quantum tunnelling occurs physically (not simulated),
eliminating the iterative barrier-crossing process. The ~4× speedup estimate is conservative
vs theoretical quantum annealing advantage (Kadowaki & Nishimori 1998).

---

## Summary Table

| Hardware | Type | Tier | FP32 TFLOPS | INT8 TOPS | TDP |
|---|---|---|---|---|---|
| RTX 3070 Laptop | GPU | 1 | 15.0 | 36.0 | 80.0W |
| Ryzen 7 6800H (CPU-only) | CPU | 1 | 0.384 | 0.768 | 45.0W |
| NVIDIA A100 80GB | GPU | 2 | 312.0 | 624.0 | 400.0W |
| NVIDIA T4 (Cloud) | GPU | 2 | 65.0 | 130.0 | 70.0W |
| AMD EPYC 7763 (Server CPU) | CPU | 2 | 2.765 | 5.53 | 280.0W |
| IBM Eagle (127 qubits) | QUANTUM | 3 | N/A | N/A | 25000.0W |
| Google Sycamore (53 qubits) | QUANTUM | 3 | N/A | N/A | 25000.0W |

---

## Methodology Notes

1. **Sub-linear throughput scaling (exponent 0.75):** Raw TFLOPS ratios overestimate real
   speedup because memory bandwidth, PCIe latency, and kernel launch overhead don't scale
   linearly. The 0.75 exponent is derived from MLPerf Inference v3.1 cross-hardware results.

2. **Energy projection:** Projected energy = (target TDP × target utilisation) × projected time.
   Server GPUs assumed at 85% utilisation; laptop GPU at 60%; CPUs at 70% and 50% respectively.

3. **Quantum energy:** Dominated by cryogenic cooling (~25 kW). Quantum processor active
   power estimated at ~5W (IBM Quantum system power budgets).

4. **Accuracy is hardware-independent:** Model weights and inference logic are identical
   across hardware — only throughput and energy change.

*All projections are estimates for academic comparison purposes.*