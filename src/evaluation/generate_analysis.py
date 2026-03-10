# generate_analysis.py — Produce formatted comparison tables for the dissertation
#
# Usage:
#     python -m src.evaluation.generate_analysis
#
# Outputs:
#   - Printed tables to console (copy-paste into report)
#   - analysis/tables/*.csv  (machine-readable)
#   - analysis/tables/*.txt  (formatted text tables)

"""
Generates structured comparisons across all three architectures
(Custom CNN, ResNet-18, VGG-16) and all optimisation techniques
(pruning, quantization, quantum-inspired), ready for dissertation write-up.
"""

import os
import sys
import pandas as pd
import numpy as np

CSV_PATH = "./data/results/inference_benchmark.csv"
OUT_DIR = "./analysis/tables"


# ======================================================================
# Helpers
# ======================================================================

def get_architecture(name: str) -> str:
    if name.startswith("RN18"):  return "ResNet-18"
    if name.startswith("VGG16"): return "VGG-16"
    return "Custom CNN"


def get_technique(name: str) -> str:
    """Map model name to a short optimisation technique label."""
    n = name
    if "Baseline" in n:                           return "Baseline"
    if "QIGA" in n:                               return "QIGA"
    if "QISA" in n:                               return "QISA"
    if "Struct" in n:
        for tok in n.split():
            if "%" in tok: return f"Struct Prune {tok}"
        return "Struct Prune"
    if "Dyn Quant" in n and "Pruned" in n:
        for tok in n.split():
            if "%" in tok: return f"Prune {tok} + DQ"
        return "Prune + DQ"
    if "Stat Quant" in n and "Pruned" in n:
        for tok in n.split():
            if "%" in tok: return f"Prune {tok} + SQ"
        return "Prune + SQ"
    if "Dynamic" in n or "Dyn Quant" in n:        return "Dyn Quant"
    if "Static" in n or "Stat Quant" in n:        return "Stat Quant"
    if "Pruned" in n:
        for tok in n.split():
            if "%" in tok: return f"Prune {tok}"
        return "Pruned"
    return name


def fmt(val, decimals=2):
    """Format a number with commas and fixed decimals."""
    if abs(val) >= 1000:
        return f"{val:,.{decimals}f}"
    return f"{val:.{decimals}f}"


def separator(char="=", width=90):
    return char * width


def section(title):
    print(f"\n{separator()}")
    print(f"  {title}")
    print(separator())


# ======================================================================
# Table 1: Full results (all models)
# ======================================================================

def table_full_results(df):
    section("TABLE 1: Full Benchmark Results — All Models")

    display = df[["model", "accuracy_pct", "throughput_imgs_per_s",
                  "total_energy_joules", "model_size_mb",
                  "avg_batch_latency_ms", "emissions_gco2"]].copy()
    display.columns = ["Model", "Acc (%)", "Throughput (img/s)",
                       "Energy (J)", "Size (MB)", "Latency (ms)", "CO₂ (g)"]

    print(display.to_string(index=False))
    display.to_csv(os.path.join(OUT_DIR, "full_results.csv"), index=False)
    return display


# ======================================================================
# Table 2: Architecture baseline comparison
# ======================================================================

def table_architecture_baselines(df):
    section("TABLE 2: Architecture Baseline Comparison")

    baselines = df[df["model"].isin(
        ["Baseline (FP32)", "RN18 Baseline", "VGG16 Baseline"]
    )].copy()
    baselines["Architecture"] = [get_architecture(m) for m in baselines["model"]]

    cols = {
        "Architecture": "Architecture",
        "accuracy_pct": "Accuracy (%)",
        "throughput_imgs_per_s": "Throughput (img/s)",
        "total_energy_joules": "Energy (J)",
        "model_size_mb": "Size (MB)",
        "avg_batch_latency_ms": "Latency (ms)",
        "emissions_gco2": "CO₂ (g)",
    }
    display = baselines[list(cols.keys())].copy()
    display.columns = list(cols.values())
    print(display.to_string(index=False))
    display.to_csv(os.path.join(OUT_DIR, "architecture_baselines.csv"), index=False)
    return display


# ======================================================================
# Table 3: Optimisation technique comparison (cross-architecture)
# ======================================================================

def table_cross_architecture(df):
    section("TABLE 3: Cross-Architecture Optimisation Comparison")
    print("  Shows accuracy for each technique across architectures.\n")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["technique"] = [get_technique(m) for m in df2["model"]]

    # Only keep techniques present in multiple architectures
    common = ["Baseline", "Prune 20%", "Prune 40%", "Prune 60%",
              "Dyn Quant", "QIGA", "QISA"]

    rows = []
    for tech in common:
        row = {"Technique": tech}
        for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
            match = df2[(df2["arch"] == arch) & (df2["technique"] == tech)]
            if not match.empty:
                row[f"{arch} Acc (%)"] = f"{match.iloc[0]['accuracy_pct']:.2f}"
                row[f"{arch} Energy (J)"] = f"{match.iloc[0]['total_energy_joules']:.1f}"
                row[f"{arch} Throughput"] = f"{match.iloc[0]['throughput_imgs_per_s']:.0f}"
            else:
                row[f"{arch} Acc (%)"] = "—"
                row[f"{arch} Energy (J)"] = "—"
                row[f"{arch} Throughput"] = "—"
        rows.append(row)

    display = pd.DataFrame(rows)
    print(display.to_string(index=False))
    display.to_csv(os.path.join(OUT_DIR, "cross_architecture_comparison.csv"), index=False)
    return display


# ======================================================================
# Table 4: Best model per architecture
# ======================================================================

def table_best_per_architecture(df):
    section("TABLE 4: Best Model per Architecture")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["technique"] = [get_technique(m) for m in df2["model"]]

    rows = []
    for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
        subset = df2[df2["arch"] == arch]
        # Best accuracy
        best_acc = subset.loc[subset["accuracy_pct"].idxmax()]
        # Most energy efficient (lowest energy)
        best_energy = subset.loc[subset["total_energy_joules"].idxmin()]
        # Best throughput
        best_tp = subset.loc[subset["throughput_imgs_per_s"].idxmax()]

        rows.append({
            "Architecture": arch,
            "Metric": "Highest Accuracy",
            "Model": best_acc["model"],
            "Technique": best_acc["technique"],
            "Value": f"{best_acc['accuracy_pct']:.2f}%",
        })
        rows.append({
            "Architecture": arch,
            "Metric": "Lowest Energy",
            "Model": best_energy["model"],
            "Technique": best_energy["technique"],
            "Value": f"{best_energy['total_energy_joules']:.1f} J",
        })
        rows.append({
            "Architecture": arch,
            "Metric": "Highest Throughput",
            "Model": best_tp["model"],
            "Technique": best_tp["technique"],
            "Value": f"{best_tp['throughput_imgs_per_s']:.0f} img/s",
        })

    display = pd.DataFrame(rows)
    print(display.to_string(index=False))
    display.to_csv(os.path.join(OUT_DIR, "best_per_architecture.csv"), index=False)
    return display


# ======================================================================
# Table 5: Pruning effectiveness
# ======================================================================

def table_pruning_comparison(df):
    section("TABLE 5: Pruning Effectiveness Across Architectures")
    print("  Shows accuracy change from baseline at each pruning level.\n")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["technique"] = [get_technique(m) for m in df2["model"]]

    # Get baselines
    baselines = {}
    for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
        bl = df2[(df2["arch"] == arch) & (df2["technique"] == "Baseline")]
        if not bl.empty:
            baselines[arch] = bl.iloc[0]["accuracy_pct"]

    rows = []
    for level in ["Prune 20%", "Prune 40%", "Prune 60%", "Prune 80%"]:
        row = {"Pruning Level": level}
        for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
            match = df2[(df2["arch"] == arch) & (df2["technique"] == level)]
            if not match.empty and arch in baselines:
                acc = match.iloc[0]["accuracy_pct"]
                delta = acc - baselines[arch]
                sign = "+" if delta >= 0 else ""
                row[f"{arch} Acc (%)"] = f"{acc:.2f}"
                row[f"{arch} Δ"] = f"{sign}{delta:.2f}"
            else:
                row[f"{arch} Acc (%)"] = "—"
                row[f"{arch} Δ"] = "—"
        rows.append(row)

    display = pd.DataFrame(rows)
    print(display.to_string(index=False))
    display.to_csv(os.path.join(OUT_DIR, "pruning_comparison.csv"), index=False)
    return display


# ======================================================================
# Table 6: Quantum-inspired optimisation results
# ======================================================================

def table_quantum_comparison(df):
    section("TABLE 6: Quantum-Inspired Optimisation Results")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["technique"] = [get_technique(m) for m in df2["model"]]

    # Get baselines
    baselines = {}
    for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
        bl = df2[(df2["arch"] == arch) & (df2["technique"] == "Baseline")]
        if not bl.empty:
            baselines[arch] = bl.iloc[0]

    rows = []
    for tech in ["QIGA", "QISA"]:
        for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
            match = df2[(df2["arch"] == arch) & (df2["technique"] == tech)]
            if not match.empty and arch in baselines:
                m = match.iloc[0]
                bl = baselines[arch]
                acc_delta = m["accuracy_pct"] - bl["accuracy_pct"]
                energy_delta = ((m["total_energy_joules"] - bl["total_energy_joules"])
                                / bl["total_energy_joules"] * 100)
                sign_a = "+" if acc_delta >= 0 else ""
                sign_e = "+" if energy_delta >= 0 else ""
                rows.append({
                    "Technique": tech,
                    "Architecture": arch,
                    "Acc (%)": f"{m['accuracy_pct']:.2f}",
                    "Acc Δ": f"{sign_a}{acc_delta:.2f}",
                    "Energy (J)": f"{m['total_energy_joules']:.1f}",
                    "Energy Δ": f"{sign_e}{energy_delta:.1f}%",
                    "Throughput": f"{m['throughput_imgs_per_s']:.0f}",
                })

    display = pd.DataFrame(rows)
    print(display.to_string(index=False))
    display.to_csv(os.path.join(OUT_DIR, "quantum_comparison.csv"), index=False)
    return display


# ======================================================================
# Table 7: Energy efficiency ranking
# ======================================================================

def table_energy_efficiency(df):
    section("TABLE 7: Energy Efficiency Ranking (Accuracy per Joule)")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["efficiency"] = df2["accuracy_pct"] / df2["total_energy_joules"]

    # Top 10 most efficient
    top = df2.nlargest(10, "efficiency")
    display = top[["model", "arch", "accuracy_pct", "total_energy_joules", "efficiency"]].copy()
    display.columns = ["Model", "Architecture", "Acc (%)", "Energy (J)", "Acc/Joule"]
    display["Acc/Joule"] = display["Acc/Joule"].apply(lambda x: f"{x:.4f}")
    print(display.to_string(index=False))

    # Full ranking
    full = df2.sort_values("efficiency", ascending=False)
    full_out = full[["model", "arch", "accuracy_pct", "total_energy_joules", "efficiency"]].copy()
    full_out.columns = ["Model", "Architecture", "Acc (%)", "Energy (J)", "Acc/Joule"]
    full_out.to_csv(os.path.join(OUT_DIR, "energy_efficiency_ranking.csv"), index=False)
    return display


# ======================================================================
# Table 8: Summary statistics per architecture
# ======================================================================

def table_summary_stats(df):
    section("TABLE 8: Summary Statistics per Architecture")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]

    stats = df2.groupby("arch").agg(
        Models=("model", "count"),
        Acc_Mean=("accuracy_pct", "mean"),
        Acc_Std=("accuracy_pct", "std"),
        Acc_Min=("accuracy_pct", "min"),
        Acc_Max=("accuracy_pct", "max"),
        Energy_Mean=("total_energy_joules", "mean"),
        Energy_Min=("total_energy_joules", "min"),
        Energy_Max=("total_energy_joules", "max"),
        Throughput_Mean=("throughput_imgs_per_s", "mean"),
        Size_MB=("model_size_mb", "first"),
    ).reset_index()

    stats.columns = ["Architecture", "# Models", "Acc Mean", "Acc Std",
                     "Acc Min", "Acc Max", "Energy Mean (J)", "Energy Min (J)",
                     "Energy Max (J)", "Throughput Mean", "Size (MB)"]

    for col in ["Acc Mean", "Acc Std", "Acc Min", "Acc Max"]:
        stats[col] = stats[col].apply(lambda x: f"{x:.2f}")
    for col in ["Energy Mean (J)", "Energy Min (J)", "Energy Max (J)"]:
        stats[col] = stats[col].apply(lambda x: f"{x:.1f}")
    stats["Throughput Mean"] = stats["Throughput Mean"].apply(lambda x: f"{x:.0f}")

    print(stats.to_string(index=False))
    stats.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"), index=False)
    return stats


# ======================================================================
# Key findings summary
# ======================================================================

def print_key_findings(df):
    section("KEY FINDINGS FOR DISCUSSION")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["technique"] = [get_technique(m) for m in df2["model"]]
    df2["efficiency"] = df2["accuracy_pct"] / df2["total_energy_joules"]

    # Overall best accuracy
    best_acc = df2.loc[df2["accuracy_pct"].idxmax()]
    print(f"\n  1. HIGHEST ACCURACY overall:")
    print(f"     {best_acc['model']} — {best_acc['accuracy_pct']:.2f}%")

    # Overall most energy efficient
    best_eff = df2.loc[df2["efficiency"].idxmax()]
    print(f"\n  2. MOST ENERGY-EFFICIENT model (accuracy per joule):")
    print(f"     {best_eff['model']} — {best_eff['efficiency']:.4f} acc/J")

    # Overall lowest energy
    low_e = df2.loc[df2["total_energy_joules"].idxmin()]
    print(f"\n  3. LOWEST ENERGY consumption:")
    print(f"     {low_e['model']} — {low_e['total_energy_joules']:.1f} J")

    # Overall fastest
    fast = df2.loc[df2["throughput_imgs_per_s"].idxmax()]
    print(f"\n  4. HIGHEST THROUGHPUT:")
    print(f"     {fast['model']} — {fast['throughput_imgs_per_s']:.0f} img/s")

    # Pruning effect
    print(f"\n  5. PRUNING REGULARISATION EFFECT:")
    for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
        sub = df2[df2["arch"] == arch]
        bl = sub[sub["technique"] == "Baseline"]
        pruned = sub[sub["technique"].str.startswith("Prune") &
                     ~sub["technique"].str.contains("\\+")]
        if not bl.empty and not pruned.empty:
            bl_acc = bl.iloc[0]["accuracy_pct"]
            best_p = pruned.loc[pruned["accuracy_pct"].idxmax()]
            delta = best_p["accuracy_pct"] - bl_acc
            sign = "+" if delta >= 0 else ""
            print(f"     {arch}: baseline {bl_acc:.2f}% → best pruned "
                  f"{best_p['accuracy_pct']:.2f}% ({sign}{delta:.2f}%) "
                  f"[{best_p['technique']}]")

    # Quantum vs classical pruning
    print(f"\n  6. QUANTUM-INSPIRED vs CLASSICAL PRUNING:")
    for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
        sub = df2[df2["arch"] == arch]
        for qt in ["QIGA", "QISA"]:
            qm = sub[sub["technique"] == qt]
            if not qm.empty:
                q_acc = qm.iloc[0]["accuracy_pct"]
                # Compare to best classical pruning
                pruned = sub[sub["technique"].str.startswith("Prune") &
                             ~sub["technique"].str.contains("\\+")]
                if not pruned.empty:
                    best_p = pruned.loc[pruned["accuracy_pct"].idxmax()]
                    delta = q_acc - best_p["accuracy_pct"]
                    sign = "+" if delta >= 0 else ""
                    print(f"     {arch} {qt}: {q_acc:.2f}% vs best prune "
                          f"{best_p['accuracy_pct']:.2f}% ({sign}{delta:.2f}%)")

    # Dynamic quantization cost
    print(f"\n  7. DYNAMIC QUANTIZATION OVERHEAD:")
    for arch in ["Custom CNN", "ResNet-18", "VGG-16"]:
        sub = df2[df2["arch"] == arch]
        bl = sub[sub["technique"] == "Baseline"]
        dq = sub[sub["technique"] == "Dyn Quant"]
        if not bl.empty and not dq.empty:
            bl_tp = bl.iloc[0]["throughput_imgs_per_s"]
            dq_tp = dq.iloc[0]["throughput_imgs_per_s"]
            slowdown = bl_tp / dq_tp
            bl_e = bl.iloc[0]["total_energy_joules"]
            dq_e = dq.iloc[0]["total_energy_joules"]
            print(f"     {arch}: {slowdown:.0f}x slower, "
                  f"{dq_e/bl_e:.1f}x more energy")

    # Carbon footprint
    total_co2 = df2["emissions_gco2"].sum()
    print(f"\n  8. TOTAL CARBON FOOTPRINT of all benchmarks:")
    print(f"     {total_co2:.4f} gCO₂  ({total_co2*1000:.2f} mgCO₂)")

    print()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print(separator("="))
    print("  BENCHMARK ANALYSIS — DISSERTATION COMPARISON TABLES")
    print(separator("="))

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run the benchmark first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} models from {CSV_PATH}\n")

    table_full_results(df)
    table_architecture_baselines(df)
    table_cross_architecture(df)
    table_best_per_architecture(df)
    table_pruning_comparison(df)
    table_quantum_comparison(df)
    table_energy_efficiency(df)
    table_summary_stats(df)
    print_key_findings(df)

    print(separator("="))
    print(f"  Tables saved to {OUT_DIR}/")
    print(f"  CSV files can be imported into LaTeX/Word")
    print(separator("="))


if __name__ == "__main__":
    main()
