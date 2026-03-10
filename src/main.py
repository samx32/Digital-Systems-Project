# main.py - Run the full experimental pipeline
#
# Usage:
#     python -m src.main
#
# This executes every stage in order:
#   1. Train baseline CIFAR-10 CNN
#   2. Unstructured pruning (20/40/60/80%) + fine-tune
#   3. Structured pruning (20/40/60%) + fine-tune
#   4. Dynamic & static quantization
#   5. Combined pruning + quantization
#   6. QIGA & QISA quantum-inspired optimization (custom CNN)
#   7. Train ResNet-18 baseline (pretrained, fine-tuned on CIFAR-10)
#   8. Train VGG-16 baseline (pretrained, fine-tuned on CIFAR-10)
#   9. Pretrained classical optimization (pruning + dynamic quant)
#  10. Pretrained quantum-inspired optimization (QIGA + QISA)
#  11. Inference benchmark (all models compared side-by-side)

import importlib
import time
import sys


STAGES = [
    ("1/11  Baseline training",                "src.baseline.cifar10_training_baseline"),
    ("2/11  Unstructured pruning",             "src.classical_optimisation.cifar10_pruning"),
    ("3/11  Structured pruning",               "src.classical_optimisation.cifar10_structured_pruning"),
    ("4/11  Quantization",                     "src.classical_optimisation.cifar10_quantization"),
    ("5/11  Combined pruning + quantization",  "src.classical_optimisation.cifar10_combined_optimization"),
    ("6/11  Quantum-inspired optimization",    "src.quantum_inspired.cifar10_quantum_optimization"),
    ("7/11  ResNet-18 baseline training",      "src.baseline.resnet18_cifar10"),
    ("8/11  VGG-16 baseline training",         "src.baseline.vgg16_cifar10"),
    ("9/11  Pretrained classical optimization","src.classical_optimisation.pretrained_optimization"),
    ("10/11 Pretrained quantum optimization",  "src.quantum_inspired.pretrained_quantum_optimization"),
    ("11/11 Inference benchmark",              "src.evaluation.inference_benchmark"),
]


def main():
    print("=" * 70)
    print("  FULL EXPERIMENTAL PIPELINE")
    print("=" * 70)
    print()

    overall_start = time.time()

    for label, module_name in STAGES:
        print()
        print("#" * 70)
        print(f"  STAGE {label}")
        print(f"  Module: {module_name}")
        print("#" * 70)
        print()

        stage_start = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception as e:
            print(f"\n  [ERROR] Stage failed: {e}")
            print(f"  Continuing to next stage...\n")
        stage_elapsed = time.time() - stage_start
        print(f"\n  Stage completed in {stage_elapsed:.1f}s")

    overall_elapsed = time.time() - overall_start
    minutes = int(overall_elapsed // 60)
    seconds = overall_elapsed % 60

    print()
    print("=" * 70)
    print(f"  ALL STAGES COMPLETE — Total time: {minutes}m {seconds:.1f}s")
    print("  Results saved to data/results/")
    print("  Models saved to data/models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
