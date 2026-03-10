# generate_graphs.py - Benchmark Comparison Graphs & Algorithm Animations

"""
Visualisation Script

Reads inference_benchmark.csv and generates:
  1. Bar charts — accuracy, throughput, energy, model size, latency, emissions
  2. Scatter plots — accuracy vs energy (Pareto), accuracy vs throughput
  3. Grouped category comparison (pruning vs quantization vs quantum)
  4. Radar chart — multi-metric comparison of best from each category
  5. Stacked energy breakdown (CPU vs GPU)
  6. Algorithm animations — QIGA convergence, QISA cooling, pruning effect

Usage:
    python -m src.evaluation.generate_graphs
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style — white background, consistent across all graphs
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "#cccccc",
    "axes.grid":         False,
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH = "./data/results/inference_benchmark.csv"
GRAPHS_DIR = "./analysis/graphs"
ANIM_DIR = "./analysis/graphs/animations"
DPI = 180
FIGSIZE = (14, 10)
FIGSIZE_WIDE = (16, 8)

# Colour palette — one colour per optimisation category
CATEGORY_COLOURS = {
    # Custom CNN
    "Baseline":          "#2196F3",
    "Unstructured":      "#4CAF50",
    "Structured":        "#FF9800",
    "Dynamic Quant":     "#F44336",
    "Static Quant":      "#9C27B0",
    "Combined (Dyn)":    "#E91E63",
    "Combined (Stat)":   "#673AB7",
    "Quantum (QIGA)":    "#00BCD4",
    "Quantum (QISA)":    "#009688",
    # ResNet-18
    "RN18 Baseline":     "#1565C0",
    "RN18 Pruned":       "#2E7D32",
    "RN18 Dyn Quant":    "#C62828",
    "RN18 QIGA":         "#00838F",
    "RN18 QISA":         "#00695C",
    # VGG-16
    "VGG16 Baseline":    "#6A1B9A",
    "VGG16 Pruned":      "#F57F17",
    "VGG16 Dyn Quant":   "#AD1457",
    "VGG16 QIGA":        "#0277BD",
    "VGG16 QISA":        "#558B2F",
}

# Architecture grouping for cross-architecture charts
ARCH_COLOURS = {
    "Custom CNN":  "#2196F3",
    "ResNet-18":   "#E53935",
    "VGG-16":      "#7B1FA2",
}

def get_architecture(name: str) -> str:
    """Return the architecture family for a model name."""
    if name.startswith("RN18"):    return "ResNet-18"
    if name.startswith("VGG16"):   return "VGG-16"
    return "Custom CNN"

def categorise(name: str) -> str:
    """Assign a category to each model name."""
    # --- ResNet-18 ---
    if name.startswith("RN18"):
        if "Baseline" in name:     return "RN18 Baseline"
        if "QIGA" in name:         return "RN18 QIGA"
        if "QISA" in name:         return "RN18 QISA"
        if "Dyn Quant" in name:    return "RN18 Dyn Quant"
        if "Pruned" in name:       return "RN18 Pruned"
        return "RN18 Baseline"
    # --- VGG-16 ---
    if name.startswith("VGG16"):
        if "Baseline" in name:     return "VGG16 Baseline"
        if "QIGA" in name:         return "VGG16 QIGA"
        if "QISA" in name:         return "VGG16 QISA"
        if "Dyn Quant" in name:    return "VGG16 Dyn Quant"
        if "Pruned" in name:       return "VGG16 Pruned"
        return "VGG16 Baseline"
    # --- Custom CNN ---
    if "Baseline" in name:          return "Baseline"
    if "Struct" in name:            return "Structured"
    if "Dyn Quant" in name:         return "Combined (Dyn)"
    if "Stat Quant" in name:        return "Combined (Stat)"
    if "Dynamic" in name:           return "Dynamic Quant"
    if "Static" in name:            return "Static Quant"
    if "QIGA" in name:              return "Quantum (QIGA)"
    if "QISA" in name:              return "Quantum (QISA)"
    if "Pruned" in name:            return "Unstructured"
    return "Other"


def get_colours(df: pd.DataFrame) -> list:
    """Return a list of colours matching each row's category."""
    return [CATEGORY_COLOURS.get(categorise(m), "#999999") for m in df["model"]]


def short_name(name: str) -> str:
    """Shorten model names for axis labels."""
    replacements = {
        # Custom CNN
        "Baseline (FP32)": "Baseline",
        "Dynamic Quantized INT8": "Dyn Quant",
        "Static Quantized INT8": "Stat Quant",
        "Pruned 20% + Dyn Quant": "P20+DQ",
        "Pruned 40% + Dyn Quant": "P40+DQ",
        "Pruned 60% + Dyn Quant": "P60+DQ",
        "Pruned 20% + Stat Quant": "P20+SQ",
        "Pruned 40% + Stat Quant": "P40+SQ",
        "Pruned 60% + Stat Quant": "P60+SQ",
        "Struct Pruned 20%": "Struct 20%",
        "Struct Pruned 40%": "Struct 40%",
        "Struct Pruned 60%": "Struct 60%",
        "QIGA Optimized": "QIGA",
        "QISA Optimized": "QISA",
        # ResNet-18
        "RN18 Baseline": "RN18",
        "RN18 Pruned 20%": "RN18 P20",
        "RN18 Pruned 40%": "RN18 P40",
        "RN18 Pruned 60%": "RN18 P60",
        "RN18 Dyn Quant": "RN18 DQ",
        "RN18 QIGA": "RN18 QIGA",
        "RN18 QISA": "RN18 QISA",
        # VGG-16
        "VGG16 Baseline": "VGG16",
        "VGG16 Pruned 20%": "VGG16 P20",
        "VGG16 Pruned 40%": "VGG16 P40",
        "VGG16 Pruned 60%": "VGG16 P60",
        "VGG16 Dyn Quant": "VGG16 DQ",
        "VGG16 QIGA": "VGG16 QIGA",
        "VGG16 QISA": "VGG16 QISA",
    }
    return replacements.get(name, name)


# ======================================================================
# 1. BAR CHARTS
# ======================================================================

def bar_chart(df, column, ylabel, title, filename, higher_is_better=True, fmt=".2f"):
    """Generic horizontal bar chart sorted by metric."""
    sorted_df = df.sort_values(column, ascending=not higher_is_better)
    colours = get_colours(sorted_df)
    labels = [short_name(m) for m in sorted_df["model"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(labels, sorted_df[column], color=colours, edgecolor="white", linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars, sorted_df[column]):
        ax.text(bar.get_width() + max(sorted_df[column]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:{fmt}}", va="center", fontsize=8)

    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    sns.despine(left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, filename), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def generate_bar_charts(df):
    """Generate all individual bar charts."""
    print("\nGenerating bar charts...")
    bar_chart(df, "accuracy_pct", "Accuracy (%)", "Model Accuracy Comparison",
              "bar_accuracy.png", higher_is_better=True, fmt=".2f")
    bar_chart(df, "throughput_imgs_per_s", "Throughput (images/sec)", "Inference Throughput Comparison",
              "bar_throughput.png", higher_is_better=True, fmt=".0f")
    bar_chart(df, "total_energy_joules", "Total Energy (Joules)", "Energy Consumption Comparison",
              "bar_energy.png", higher_is_better=False, fmt=".1f")
    bar_chart(df, "model_size_mb", "Model Size (MB)", "Model Size Comparison",
              "bar_model_size.png", higher_is_better=False, fmt=".2f")
    bar_chart(df, "avg_batch_latency_ms", "Avg Batch Latency (ms)", "Batch Latency Comparison",
              "bar_latency.png", higher_is_better=False, fmt=".2f")
    bar_chart(df, "emissions_gco2", "CO₂ Emissions (g)", "Carbon Emissions Comparison",
              "bar_emissions.png", higher_is_better=False, fmt=".4f")


# ======================================================================
# 2. SCATTER / TRADE-OFF PLOTS
# ======================================================================

def scatter_accuracy_vs_energy(df):
    """Accuracy vs Energy scatter — shows Pareto trade-off."""
    print("  Generating accuracy vs energy scatter...")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colours = get_colours(df)

    for i, row in df.iterrows():
        ax.scatter(row["total_energy_joules"], row["accuracy_pct"],
                   c=CATEGORY_COLOURS.get(categorise(row["model"]), "#999"),
                   s=120, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(short_name(row["model"]),
                    (row["total_energy_joules"], row["accuracy_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    # Draw Pareto frontier
    pareto = df.sort_values("total_energy_joules")
    best_acc = -1
    pareto_pts = []
    for _, row in pareto.iterrows():
        if row["accuracy_pct"] > best_acc:
            best_acc = row["accuracy_pct"]
            pareto_pts.append((row["total_energy_joules"], row["accuracy_pct"]))
    if pareto_pts:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, "--", color="#888", linewidth=1, alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("Total Energy (Joules)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Energy Trade-off", fontsize=13, fontweight="bold")

    # Legend by category
    handles = []
    for cat, col in CATEGORY_COLOURS.items():
        if any(categorise(m) == cat for m in df["model"]):
            handles.append(plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=col, markersize=8, label=cat))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "scatter_accuracy_vs_energy.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_accuracy_vs_energy.png")


def scatter_accuracy_vs_throughput(df):
    """Accuracy vs Throughput scatter."""
    print("  Generating accuracy vs throughput scatter...")
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for _, row in df.iterrows():
        ax.scatter(row["throughput_imgs_per_s"], row["accuracy_pct"],
                   c=CATEGORY_COLOURS.get(categorise(row["model"]), "#999"),
                   s=120, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(short_name(row["model"]),
                    (row["throughput_imgs_per_s"], row["accuracy_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Throughput (images/sec)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Throughput", fontsize=13, fontweight="bold")

    handles = []
    for cat, col in CATEGORY_COLOURS.items():
        if any(categorise(m) == cat for m in df["model"]):
            handles.append(plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=col, markersize=8, label=cat))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "scatter_accuracy_vs_throughput.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_accuracy_vs_throughput.png")


# ======================================================================
# 3. GROUPED CATEGORY COMPARISON
# ======================================================================

def grouped_category_comparison(df):
    """Side-by-side grouped bars: best of each category for key metrics."""
    print("  Generating category comparison...")

    # Pick the best model (highest accuracy) from each category
    df_cat = df.copy()
    df_cat["category"] = [categorise(m) for m in df_cat["model"]]
    best = df_cat.loc[df_cat.groupby("category")["accuracy_pct"].idxmax()]
    best = best.sort_values("accuracy_pct", ascending=False)

    metrics = ["accuracy_pct", "throughput_imgs_per_s", "total_energy_joules", "model_size_mb"]
    metric_labels = ["Accuracy (%)", "Throughput (img/s)", "Energy (J)", "Size (MB)"]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE)
    axes = axes.flatten()

    for ax, metric, label in zip(axes, metrics, metric_labels):
        colours = [CATEGORY_COLOURS.get(cat, "#999") for cat in best["category"]]
        cat_labels = best["category"].values
        vals = best[metric].values

        bars = ax.bar(range(len(vals)), vals, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        sns.despine(ax=ax)

    fig.suptitle("Best Model per Category — Key Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "grouped_category_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved grouped_category_comparison.png")


# ======================================================================
# 4. RADAR CHART
# ======================================================================

def radar_chart(df):
    """Radar chart comparing best model from each category across 5 metrics."""
    print("  Generating radar chart...")

    df_cat = df.copy()
    df_cat["category"] = [categorise(m) for m in df_cat["model"]]
    best = df_cat.loc[df_cat.groupby("category")["accuracy_pct"].idxmax()]

    metrics = ["accuracy_pct", "throughput_imgs_per_s", "model_size_mb",
               "total_energy_joules", "avg_batch_latency_ms"]
    display_labels = ["Accuracy", "Throughput", "Small Size", "Low Energy", "Low Latency"]

    # Normalise 0–1, invert "lower is better" metrics
    normalised = pd.DataFrame()
    for m in metrics:
        col = best[m].values.astype(float)
        mn, mx = col.min(), col.max()
        if mx == mn:
            norm = np.ones_like(col)
        else:
            norm = (col - mn) / (mx - mn)
        # For size, energy, latency: lower is better → invert
        if m in ("model_size_mb", "total_energy_joules", "avg_batch_latency_ms"):
            norm = 1 - norm
        normalised[m] = norm

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for idx, (_, row) in enumerate(best.iterrows()):
        values = normalised.iloc[idx].values.tolist()
        values += values[:1]
        cat = row["category"]
        colour = CATEGORY_COLOURS.get(cat, "#999")
        ax.plot(angles, values, "o-", linewidth=1.5, label=cat, color=colour)
        ax.fill(angles, values, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Metric Radar — Best per Category\n(higher = better)", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "radar_chart.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved radar_chart.png")


# ======================================================================
# 5. STACKED ENERGY BREAKDOWN
# ======================================================================

def stacked_energy_chart(df):
    """Stacked bar chart showing CPU vs GPU energy per model."""
    print("  Generating stacked energy chart...")

    sorted_df = df.sort_values("total_energy_joules", ascending=True)
    labels = [short_name(m) for m in sorted_df["model"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y = range(len(labels))

    ax.barh(y, sorted_df["cpu_energy_joules"], color="#FF7043", label="CPU Energy", edgecolor="white", linewidth=0.5)
    ax.barh(y, sorted_df["gpu_energy_joules"], left=sorted_df["cpu_energy_joules"].values,
            color="#42A5F5", label="GPU Energy", edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Energy (Joules)", fontsize=11)
    ax.set_title("CPU vs GPU Energy Breakdown", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    sns.despine(left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "stacked_energy_breakdown.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved stacked_energy_breakdown.png")


# ======================================================================
# 7. ARCHITECTURE COMPARISON CHARTS
# ======================================================================

def architecture_baseline_comparison(df):
    """Side-by-side grouped bars: baseline of each architecture across key metrics."""
    print("  Generating architecture baseline comparison...")

    baselines = df[df["model"].apply(
        lambda m: m in ("Baseline (FP32)", "RN18 Baseline", "VGG16 Baseline")
    )].copy()
    if baselines.empty:
        print("  [SKIP] No baseline models found for all three architectures.")
        return

    baselines["arch"] = [get_architecture(m) for m in baselines["model"]]

    metrics = ["accuracy_pct", "throughput_imgs_per_s", "total_energy_joules", "model_size_mb"]
    metric_labels = ["Accuracy (%)", "Throughput (img/s)", "Energy (J)", "Size (MB)"]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE)
    axes = axes.flatten()

    for ax, metric, label in zip(axes, metrics, metric_labels):
        colours = [ARCH_COLOURS.get(a, "#999") for a in baselines["arch"]]
        bars = ax.bar(baselines["arch"], baselines[metric], color=colours,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, baselines[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        sns.despine(ax=ax)

    fig.suptitle("Architecture Baseline Comparison\nCustom CNN vs ResNet-18 vs VGG-16",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "arch_baseline_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved arch_baseline_comparison.png")


def architecture_optimisation_heatmap(df):
    """Heatmap: accuracy across architectures × optimisation techniques."""
    print("  Generating architecture optimisation heatmap...")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    df2["opt"] = [categorise(m) for m in df2["model"]]

    # Build a mapping from (arch, opt_short) to accuracy
    # Normalise opt names so the same technique across architectures shares a row
    def opt_short(model_name):
        n = model_name
        if "Baseline" in n or n in ("Baseline (FP32)", "RN18 Baseline", "VGG16 Baseline"):
            return "Baseline"
        if "QIGA" in n:         return "QIGA"
        if "QISA" in n:         return "QISA"
        if "Dyn Quant" in n and "Pruned" not in n:
            return "Dyn Quant"
        if "Stat Quant" in n and "Pruned" not in n:
            return "Stat Quant"
        if "Struct" in n:       return n.replace("Struct Pruned ", "Struct P")
        # Pruned + combined
        if "+" in n:            return n.split(") ")[-1] if ") " in n else n
        if "Pruned" in n:
            # Extract percentage
            for tok in n.split():
                if "%" in tok:
                    return f"Pruned {tok}"
        if "Dynamic" in n:      return "Dyn Quant"
        if "Static" in n:       return "Stat Quant"
        return n

    df2["opt_short"] = [opt_short(m) for m in df2["model"]]

    # Pivot
    pivot = df2.pivot_table(index="opt_short", columns="arch",
                             values="accuracy_pct", aggfunc="max")
    # Sort by mean accuracy
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("Accuracy Across Architectures \u00d7 Optimisation Techniques",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Architecture", fontsize=11)
    ax.set_ylabel("Optimisation", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "arch_optimisation_heatmap.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved arch_optimisation_heatmap.png")


def architecture_best_comparison(df):
    """Bar chart: best accuracy achieved per architecture + the technique used."""
    print("  Generating best-per-architecture comparison...")

    df2 = df.copy()
    df2["arch"] = [get_architecture(m) for m in df2["model"]]
    best = df2.loc[df2.groupby("arch")["accuracy_pct"].idxmax()]
    best = best.sort_values("accuracy_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = [ARCH_COLOURS.get(a, "#999") for a in best["arch"]]
    bars = ax.bar(best["arch"], best["accuracy_pct"], color=colours,
                  edgecolor="white", linewidth=0.5, width=0.5)

    for bar, (_, row) in zip(bars, best.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{row['accuracy_pct']:.2f}%\n({short_name(row['model'])})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Best Accuracy (%)", fontsize=11)
    ax.set_title("Best Accuracy per Architecture", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(best["accuracy_pct"]) * 1.12)
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "arch_best_accuracy.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved arch_best_accuracy.png")


def architecture_efficiency_scatter(df):
    """Scatter: accuracy vs energy for all models, marker-shaped by architecture."""
    print("  Generating architecture efficiency scatter...")

    markers = {"Custom CNN": "o", "ResNet-18": "s", "VGG-16": "D"}
    fig, ax = plt.subplots(figsize=(14, 8))

    for _, row in df.iterrows():
        arch = get_architecture(row["model"])
        ax.scatter(row["total_energy_joules"], row["accuracy_pct"],
                   c=ARCH_COLOURS.get(arch, "#999"),
                   marker=markers.get(arch, "o"),
                   s=120, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(short_name(row["model"]),
                    (row["total_energy_joules"], row["accuracy_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=6)

    ax.set_xlabel("Total Energy (Joules)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Energy — All Architectures", fontsize=13, fontweight="bold")

    handles = []
    for arch, col in ARCH_COLOURS.items():
        handles.append(plt.Line2D([0], [0], marker=markers.get(arch, "o"), color="w",
                       markerfacecolor=col, markersize=9, label=arch))
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "arch_accuracy_vs_energy.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved arch_accuracy_vs_energy.png")


# ======================================================================
# 8. ALGORITHM ANIMATIONS
# ======================================================================

def animate_qiga_convergence():
    """
    Animated GIF showing how a Quantum-Inspired Genetic Algorithm converges.
    Simulates a population of chromosomes being evaluated and rotated toward
    the best solution over generations.
    """
    print("  Generating QIGA convergence animation...")
    np.random.seed(42)

    # Simulate QIGA optimisation: population fitness over generations
    generations = 30
    pop_size = 15
    # Start with random fitness values, converge toward optimum
    fitness_history = []
    current_best = 0.55  # starting best accuracy
    pop_fitness = np.random.uniform(0.3, 0.6, pop_size)

    for gen in range(generations):
        # Simulate quantum rotation: population drifts toward best
        noise = np.random.normal(0, 0.02, pop_size)
        improvement = 0.015 * (1 - gen / generations)  # decreasing improvement
        pop_fitness = pop_fitness + improvement + noise
        pop_fitness = np.clip(pop_fitness, 0.1, 0.95)
        current_best = max(current_best, pop_fitness.max())
        # Quantum rotation pulls population toward best
        pop_fitness = pop_fitness + 0.05 * (current_best - pop_fitness)
        fitness_history.append(pop_fitness.copy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("QIGA: Quantum-Inspired Genetic Algorithm", fontsize=14, fontweight="bold")

    # Left: population scatter, Right: convergence curve
    best_line = []
    mean_line = []
    worst_line = []

    def update(frame):
        ax1.clear()
        ax2.clear()

        pop = fitness_history[frame]
        x_positions = np.arange(pop_size)

        # Left panel: population fitness as bar chart
        colors = plt.cm.RdYlGn((pop - 0.3) / 0.65)
        ax1.bar(x_positions, pop, color=colors, edgecolor="white", linewidth=0.5)
        ax1.axhline(y=max(pop), color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7, label=f"Best: {max(pop):.3f}")
        ax1.axhline(y=np.mean(pop), color="#2196F3", linestyle=":", linewidth=1, alpha=0.7, label=f"Mean: {np.mean(pop):.3f}")
        ax1.set_ylim(0.2, 1.0)
        ax1.set_xlabel("Chromosome Index")
        ax1.set_ylabel("Fitness (Accuracy)")
        ax1.set_title(f"Population — Generation {frame + 1}/{generations}")
        ax1.legend(fontsize=8, loc="lower right")

        # Right panel: convergence over time
        best_line.append(max(pop))
        mean_line.append(np.mean(pop))
        worst_line.append(min(pop))

        gens = range(1, len(best_line) + 1)
        ax2.fill_between(gens, worst_line, best_line, alpha=0.15, color="#4CAF50")
        ax2.plot(gens, best_line, "o-", color="#4CAF50", linewidth=2, markersize=4, label="Best")
        ax2.plot(gens, mean_line, "s-", color="#2196F3", linewidth=1.5, markersize=3, label="Mean")
        ax2.plot(gens, worst_line, "^-", color="#F44336", linewidth=1, markersize=3, label="Worst")
        ax2.set_xlim(1, generations)
        ax2.set_ylim(0.2, 1.0)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness (Accuracy)")
        ax2.set_title("Convergence Over Generations")
        ax2.legend(fontsize=8)

        for a in (ax1, ax2):
            sns.despine(ax=a)

    anim = animation.FuncAnimation(fig, update, frames=generations, interval=300, repeat=False)
    anim.save(os.path.join(ANIM_DIR, "qiga_convergence.gif"), writer="pillow", fps=3, dpi=120)
    plt.close(fig)

    # Also save a static final-frame version
    best_line.clear()
    mean_line.clear()
    worst_line.clear()
    for pop in fitness_history:
        best_line.append(max(pop))
        mean_line.append(np.mean(pop))
        worst_line.append(min(pop))

    fig2, ax = plt.subplots(figsize=FIGSIZE)
    gens = range(1, generations + 1)
    ax.fill_between(gens, worst_line, best_line, alpha=0.15, color="#4CAF50")
    ax.plot(gens, best_line, "o-", color="#4CAF50", linewidth=2, markersize=5, label="Best Fitness")
    ax.plot(gens, mean_line, "s-", color="#2196F3", linewidth=1.5, markersize=4, label="Mean Fitness")
    ax.plot(gens, worst_line, "^-", color="#F44336", linewidth=1, markersize=4, label="Worst Fitness")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fitness (Accuracy)", fontsize=11)
    ax.set_title("QIGA Convergence — Population Fitness Over Generations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    fig2.savefig(os.path.join(GRAPHS_DIR, "qiga_convergence_static.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved qiga_convergence.gif + qiga_convergence_static.png")


def animate_qisa_cooling():
    """
    Animated GIF showing QISA's simulated annealing cooling schedule.
    Shows temperature decreasing and solution quality improving, with
    occasional quantum tunnelling escaping local optima.
    """
    print("  Generating QISA cooling animation...")
    np.random.seed(42)

    # Simulate QISA cooling
    initial_temp = 1.0
    final_temp = 0.01
    cooling_rate = 0.93
    temp = initial_temp

    temps = []
    fitnesses = []
    best_fitness_history = []
    tunnelling_events = []

    current_fitness = 0.50
    best_fitness = current_fitness

    step = 0
    while temp > final_temp:
        # Simulate a step
        candidate = current_fitness + np.random.normal(0, 0.05)
        candidate = np.clip(candidate, 0.1, 0.95)

        delta = candidate - current_fitness
        # SA acceptance (maximising fitness, so accept if delta > 0)
        if delta > 0:
            current_fitness = candidate
        elif np.random.random() < np.exp(delta / temp):
            current_fitness = candidate  # Accept worse solution at high temp

        # Quantum tunnelling (10% chance)
        if np.random.random() < 0.1:
            tunnel_jump = np.random.uniform(-0.1, 0.15)
            tunnelled = current_fitness + tunnel_jump
            tunnelled = np.clip(tunnelled, 0.1, 0.95)
            if tunnelled > current_fitness:
                current_fitness = tunnelled
                tunnelling_events.append(step)

        # Gradual bias toward improvement
        current_fitness += 0.003

        best_fitness = max(best_fitness, current_fitness)
        temps.append(temp)
        fitnesses.append(current_fitness)
        best_fitness_history.append(best_fitness)

        temp *= cooling_rate
        step += 1

    total_steps = len(temps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
    fig.suptitle("QISA: Quantum-Inspired Simulated Annealing", fontsize=14, fontweight="bold")

    def update(frame):
        # Show data up to current frame (sample every N steps for smoother anim)
        idx = min(frame * 2, total_steps - 1)
        ax1.clear()
        ax2.clear()

        steps_so_far = range(idx + 1)

        # Top: Temperature curve
        ax1.plot(steps_so_far, temps[:idx + 1], color="#F44336", linewidth=2)
        ax1.fill_between(steps_so_far, temps[:idx + 1], alpha=0.1, color="#F44336")
        ax1.set_ylabel("Temperature", fontsize=10)
        ax1.set_title(f"Cooling Schedule — Step {idx + 1}/{total_steps}  |  T = {temps[idx]:.4f}", fontsize=10)
        ax1.set_xlim(0, total_steps)
        ax1.set_ylim(0, 1.1)

        # Bottom: Fitness
        ax2.plot(steps_so_far, fitnesses[:idx + 1], color="#2196F3", linewidth=1, alpha=0.6, label="Current")
        ax2.plot(steps_so_far, best_fitness_history[:idx + 1], color="#4CAF50", linewidth=2, label="Best")

        # Mark tunnelling events
        tunnels_so_far = [t for t in tunnelling_events if t <= idx]
        if tunnels_so_far:
            ax2.scatter(tunnels_so_far, [fitnesses[t] for t in tunnels_so_far],
                       color="#FF9800", marker="*", s=60, zorder=5, label="Quantum Tunnel")

        ax2.set_xlabel("Step", fontsize=10)
        ax2.set_ylabel("Fitness (Accuracy)", fontsize=10)
        ax2.set_title(f"Solution Quality — Best: {best_fitness_history[idx]:.4f}", fontsize=10)
        ax2.set_xlim(0, total_steps)
        ax2.set_ylim(0.3, 1.0)
        ax2.legend(fontsize=8, loc="lower right")

        for a in (ax1, ax2):
            sns.despine(ax=a)

    n_frames = total_steps // 2
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=80, repeat=False)
    anim.save(os.path.join(ANIM_DIR, "qisa_cooling.gif"), writer="pillow", fps=12, dpi=120)
    plt.close(fig)

    # Static version
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[1, 1])
    fig2.suptitle("QISA: Cooling Schedule & Solution Convergence", fontsize=14, fontweight="bold")

    all_steps = range(total_steps)
    ax1.plot(all_steps, temps, color="#F44336", linewidth=2)
    ax1.fill_between(all_steps, temps, alpha=0.1, color="#F44336")
    ax1.set_ylabel("Temperature", fontsize=11)
    ax1.set_title("Temperature Cooling Schedule")

    ax2.plot(all_steps, fitnesses, color="#2196F3", linewidth=0.8, alpha=0.5, label="Current Fitness")
    ax2.plot(all_steps, best_fitness_history, color="#4CAF50", linewidth=2, label="Best Fitness")
    if tunnelling_events:
        ax2.scatter(tunnelling_events, [fitnesses[t] for t in tunnelling_events],
                   color="#FF9800", marker="*", s=50, zorder=5, label="Quantum Tunnelling")
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Fitness (Accuracy)", fontsize=11)
    ax2.set_title("Solution Quality Over Time")
    ax2.legend(fontsize=9)

    for a in (ax1, ax2):
        sns.despine(ax=a)
    plt.tight_layout()
    fig2.savefig(os.path.join(GRAPHS_DIR, "qisa_cooling_static.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved qisa_cooling.gif + qisa_cooling_static.png")


def animate_pruning_effect():
    """
    Animated GIF showing how pruning progressively removes weights.
    Visualises a weight matrix going from dense to sparse.
    """
    print("  Generating pruning effect animation...")
    np.random.seed(42)

    # Create a synthetic weight matrix
    size = 32
    weights = np.random.randn(size, size)

    pruning_levels = np.linspace(0, 0.9, 30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Unstructured Pruning — Weight Removal", fontsize=14, fontweight="bold")

    def update(frame):
        ax1.clear()
        ax2.clear()

        amount = pruning_levels[frame]
        # Zero out the smallest weights
        threshold = np.percentile(np.abs(weights), amount * 100)
        pruned = weights.copy()
        pruned[np.abs(pruned) < threshold] = 0

        sparsity = np.sum(pruned == 0) / pruned.size * 100
        nonzero = np.sum(pruned != 0)

        # Left: weight heatmap
        im = ax1.imshow(pruned, cmap="RdBu_r", vmin=-3, vmax=3, aspect="equal")
        ax1.set_title(f"Weight Matrix — {amount * 100:.0f}% Pruned\n"
                      f"({nonzero}/{size * size} non-zero weights)", fontsize=10)
        ax1.set_xlabel("Output Neuron")
        ax1.set_ylabel("Input Neuron")

        # Right: weight distribution
        non_zero_weights = pruned[pruned != 0].flatten()
        if len(non_zero_weights) > 0:
            ax2.hist(non_zero_weights, bins=40, color="#4CAF50", edgecolor="white",
                     linewidth=0.3, alpha=0.8, density=True)
        ax2.axvline(x=0, color="#F44336", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(0, 0.6)
        ax2.set_xlabel("Weight Value")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Weight Distribution\nSparsity: {sparsity:.1f}%", fontsize=10)
        sns.despine(ax=ax2)

    anim = animation.FuncAnimation(fig, update, frames=len(pruning_levels), interval=200, repeat=False)
    anim.save(os.path.join(ANIM_DIR, "pruning_effect.gif"), writer="pillow", fps=5, dpi=120)
    plt.close(fig)
    print("  Saved pruning_effect.gif")


def animate_structured_pruning():
    """
    Animated GIF showing structured pruning removing entire channels.
    """
    print("  Generating structured pruning animation...")
    np.random.seed(42)

    n_channels = 16
    spatial = 8  # 8×8 feature map
    # Create synthetic feature maps for 16 channels
    feature_maps = np.random.randn(n_channels, spatial, spatial) * 0.5
    # Some channels have higher importance than others
    importance = np.abs(np.random.randn(n_channels))
    importance = importance / importance.max()

    # Sort channels by importance (lowest first to remove)
    removal_order = np.argsort(importance)

    frames_data = []
    active = np.ones(n_channels, dtype=bool)
    for step in range(n_channels - 1):
        frames_data.append(active.copy())
        active[removal_order[step]] = False
    frames_data.append(active.copy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Structured Pruning — Channel Removal", fontsize=14, fontweight="bold")

    def update(frame):
        ax1.clear()
        ax2.clear()

        active = frames_data[frame]
        n_active = active.sum()
        removed = n_channels - n_active
        pct = removed / n_channels * 100

        # Left: grid of feature maps, grey out removed channels
        grid_cols = 8
        grid_rows = 2
        combined = np.ones((grid_rows * (spatial + 1), grid_cols * (spatial + 1))) * 0.5

        for ch in range(n_channels):
            row = ch // grid_cols
            col = ch % grid_cols
            r0 = row * (spatial + 1)
            c0 = col * (spatial + 1)
            if active[ch]:
                combined[r0:r0 + spatial, c0:c0 + spatial] = feature_maps[ch]
            else:
                combined[r0:r0 + spatial, c0:c0 + spatial] = 0  # removed

        ax1.imshow(combined, cmap="viridis", vmin=-1.5, vmax=1.5, aspect="equal")
        ax1.set_title(f"Feature Map Channels — {removed} removed ({pct:.0f}%)\n"
                      f"{n_active}/{n_channels} channels active", fontsize=10)
        ax1.axis("off")

        # Right: importance bar chart
        colours = ["#4CAF50" if a else "#BDBDBD" for a in active]
        ax2.barh(range(n_channels), importance, color=colours, edgecolor="white", linewidth=0.5)
        ax2.set_yticks(range(n_channels))
        ax2.set_yticklabels([f"Ch {i}" for i in range(n_channels)], fontsize=7)
        ax2.set_xlabel("Channel Importance (L1-norm)")
        ax2.set_title(f"Channel Importance — grey = removed", fontsize=10)
        ax2.invert_yaxis()
        sns.despine(ax=ax2, left=True)

    anim = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=400, repeat=False)
    anim.save(os.path.join(ANIM_DIR, "structured_pruning.gif"), writer="pillow", fps=3, dpi=120)
    plt.close(fig)
    print("  Saved structured_pruning.gif")


def animate_quantum_tunnelling():
    """
    Animated GIF showing quantum tunnelling in optimisation.

    A 1-D fitness landscape has multiple peaks (local optima) separated by
    valleys (energy barriers).  Two agents are shown:
      - Classical SA (red) can only take small steps and gets stuck at a
        local optimum, unable to cross the barrier.
      - QISA agent (cyan) follows the same path initially but at key moments
        it "tunnels" through the barrier and reappears on the other side,
        eventually reaching the global optimum.

    The tunnelling events are drawn with a dashed arc and a starburst to
    make the teleportation visually clear.
    """
    print("  Generating quantum tunnelling animation...")
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Build a multi-modal 1-D fitness landscape
    # ------------------------------------------------------------------
    x = np.linspace(0, 10, 500)
    # Several Gaussian peaks of different heights
    landscape = (
        0.55 * np.exp(-((x - 2.0) ** 2) / 0.4) +   # local peak 1
        0.70 * np.exp(-((x - 4.5) ** 2) / 0.3) +   # local peak 2 (taller)
        0.40 * np.exp(-((x - 6.5) ** 2) / 0.5) +   # local peak 3 (small)
        1.00 * np.exp(-((x - 8.5) ** 2) / 0.5) +   # GLOBAL OPTIMUM
        0.10 * np.ones_like(x)                       # base offset
    )

    # ------------------------------------------------------------------
    # Pre-compute agent trajectories
    # ------------------------------------------------------------------
    total_frames = 90
    classical_pos = []    # Classical SA agent positions
    quantum_pos = []      # QISA agent positions
    tunnel_frames = []    # (frame, from_x, to_x) for tunnelling events

    # Both start at x = 1.0
    c_x = 1.0
    q_x = 1.0

    def landscape_at(pos):
        idx = int(np.clip(pos / 10 * (len(x) - 1), 0, len(x) - 1))
        return landscape[idx]

    c_temp = 0.8
    q_temp = 0.8

    for f in range(total_frames):
        # --- Classical SA: small random step, standard acceptance ---
        step = np.random.normal(0, 0.15)
        c_candidate = np.clip(c_x + step, 0.1, 9.9)
        delta_c = landscape_at(c_candidate) - landscape_at(c_x)
        if delta_c > 0 or np.random.random() < np.exp(delta_c / max(c_temp, 0.01)):
            c_x = c_candidate
        c_temp *= 0.97

        # --- QISA agent: same local walk + occasional tunnel ---
        step = np.random.normal(0, 0.15)
        q_candidate = np.clip(q_x + step, 0.1, 9.9)
        delta_q = landscape_at(q_candidate) - landscape_at(q_x)
        if delta_q > 0 or np.random.random() < np.exp(delta_q / max(q_temp, 0.01)):
            q_x = q_candidate

        # Quantum tunnelling at specific dramatic moments
        if f == 30 and q_x < 5.0:
            old = q_x
            q_x = 4.5 + np.random.uniform(-0.2, 0.2)  # tunnel to peak 2
            tunnel_frames.append((f, old, q_x))
        elif f == 55 and q_x < 7.5:
            old = q_x
            q_x = 8.5 + np.random.uniform(-0.3, 0.3)  # tunnel to global optimum
            tunnel_frames.append((f, old, q_x))

        q_temp *= 0.97

        classical_pos.append(c_x)
        quantum_pos.append(q_x)

    # ------------------------------------------------------------------
    # Animate
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))

    # Trail storage
    c_trail_x, c_trail_y = [], []
    q_trail_x, q_trail_y = [], []

    def update(frame):
        ax.clear()

        # Draw landscape
        ax.fill_between(x, landscape, alpha=0.12, color="#78909C")
        ax.plot(x, landscape, color="#546E7A", linewidth=2)

        # Label peaks
        ax.annotate("Local\nOptimum 1", xy=(2.0, 0.65), fontsize=7,
                    ha="center", color="#78909C", style="italic")
        ax.annotate("Local\nOptimum 2", xy=(4.5, 0.80), fontsize=7,
                    ha="center", color="#78909C", style="italic")
        ax.annotate("Local\nOptimum 3", xy=(6.5, 0.50), fontsize=7,
                    ha="center", color="#78909C", style="italic")
        ax.annotate("Global\nOptimum", xy=(8.5, 1.10), fontsize=8,
                    ha="center", color="#2E7D32", fontweight="bold")

        # Energy barriers label
        ax.annotate("", xy=(5.5, 0.18), xytext=(3.2, 0.18),
                    arrowprops=dict(arrowstyle="<->", color="#B71C1C", lw=1.2))
        ax.text(4.35, 0.20, "energy barrier", ha="center", fontsize=7,
                color="#B71C1C", style="italic")

        # --- Classical agent trail ---
        c_x_now = classical_pos[frame]
        c_y_now = landscape_at(c_x_now)
        c_trail_x.append(c_x_now)
        c_trail_y.append(c_y_now)
        if len(c_trail_x) > 1:
            ax.plot(c_trail_x, c_trail_y, "-", color="#E53935", alpha=0.25, linewidth=1)
        ax.scatter([c_x_now], [c_y_now], s=180, color="#E53935", edgecolors="white",
                   linewidth=1.5, zorder=5)
        ax.annotate("Classical SA", (c_x_now, c_y_now),
                    textcoords="offset points", xytext=(-10, 14),
                    fontsize=8, fontweight="bold", color="#E53935")

        # --- Quantum agent trail ---
        q_x_now = quantum_pos[frame]
        q_y_now = landscape_at(q_x_now)
        q_trail_x.append(q_x_now)
        q_trail_y.append(q_y_now)
        if len(q_trail_x) > 1:
            ax.plot(q_trail_x, q_trail_y, "-", color="#00ACC1", alpha=0.25, linewidth=1)
        ax.scatter([q_x_now], [q_y_now], s=180, color="#00ACC1", edgecolors="white",
                   linewidth=1.5, zorder=5)
        ax.annotate("QISA Agent", (q_x_now, q_y_now),
                    textcoords="offset points", xytext=(-10, 14),
                    fontsize=8, fontweight="bold", color="#00ACC1")

        # --- Draw tunnelling arcs for events that already happened ---
        for (tf, from_x, to_x) in tunnel_frames:
            if frame >= tf:
                # Draw a dashed arc over the barrier
                arc_x = np.linspace(from_x, to_x, 40)
                arc_height = 0.3 + 0.15 * abs(to_x - from_x)
                arc_y = (
                    landscape_at(from_x)
                    + arc_height * np.sin(np.linspace(0, np.pi, 40))
                )
                alpha = 1.0 if frame - tf < 12 else 0.35
                ax.plot(arc_x, arc_y, "--", color="#FF6F00", linewidth=2,
                        alpha=alpha, zorder=4)

                # Starburst at landing point
                if frame - tf < 8:
                    burst_size = max(200 - (frame - tf) * 30, 50)
                    ax.scatter([to_x], [landscape_at(to_x)], s=burst_size,
                              marker="*", color="#FFD600", edgecolors="#FF6F00",
                              linewidth=0.8, zorder=6)

                    # Label
                    ax.annotate("TUNNEL!",
                                (to_x, landscape_at(to_x)),
                                textcoords="offset points",
                                xytext=(12, -18),
                                fontsize=10, fontweight="bold",
                                color="#FF6F00",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc="#FFF8E1", ec="#FF6F00", alpha=0.9))

        # --- Decorations ---
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Search Space Position", fontsize=11)
        ax.set_ylabel("Fitness (Accuracy)", fontsize=11)
        ax.set_title(
            "Quantum Tunnelling in Optimisation\n"
            "Classical SA gets trapped — QISA tunnels through barriers",
            fontsize=13, fontweight="bold"
        )

        # Info box
        info = (f"Frame {frame + 1}/{total_frames}   "
                f"Classical: {landscape_at(c_x_now):.3f}   "
                f"QISA: {landscape_at(q_x_now):.3f}")
        ax.text(0.5, 0.02, info, transform=ax.transAxes, ha="center",
                fontsize=8, color="#666",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", alpha=0.8))

        sns.despine()

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=120, repeat=False
    )
    anim.save(
        os.path.join(ANIM_DIR, "quantum_tunnelling.gif"),
        writer="pillow", fps=8, dpi=130
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Static summary frame (side-by-side: stuck vs tunnelled)
    # ------------------------------------------------------------------
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle(
        "Quantum Tunnelling Explained",
        fontsize=14, fontweight="bold"
    )

    for ax_s, title, positions, colour, stuck in [
        (axL, "Classical SA — Trapped at Local Optimum",
         classical_pos, "#E53935", True),
        (axR, "QISA — Tunnels to Global Optimum",
         quantum_pos, "#00ACC1", False),
    ]:
        ax_s.fill_between(x, landscape, alpha=0.12, color="#78909C")
        ax_s.plot(x, landscape, color="#546E7A", linewidth=2)

        # Plot full trail
        trail_y = [landscape_at(p) for p in positions]
        ax_s.plot(positions, trail_y, "-", color=colour, alpha=0.2, linewidth=1)

        # Final position
        final = positions[-1]
        ax_s.scatter([final], [landscape_at(final)], s=220, color=colour,
                     edgecolors="white", linewidth=2, zorder=5)
        ax_s.annotate(
            f"Final fitness: {landscape_at(final):.3f}",
            (final, landscape_at(final)),
            textcoords="offset points", xytext=(10, 12),
            fontsize=9, fontweight="bold", color=colour,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, alpha=0.9)
        )

        # Draw tunnelling arcs on right panel
        if not stuck:
            for (_, from_x, to_x) in tunnel_frames:
                arc_x = np.linspace(from_x, to_x, 40)
                arc_height = 0.3 + 0.15 * abs(to_x - from_x)
                arc_y = (
                    landscape_at(from_x)
                    + arc_height * np.sin(np.linspace(0, np.pi, 40))
                )
                ax_s.plot(arc_x, arc_y, "--", color="#FF6F00",
                          linewidth=2, alpha=0.7)
                ax_s.scatter([to_x], [landscape_at(to_x)], marker="*",
                             s=200, color="#FFD600", edgecolors="#FF6F00",
                             linewidth=0.8, zorder=6)
                mid = len(arc_x) // 2
                ax_s.annotate(
                    "tunnel", (arc_x[mid], arc_y[mid]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, color="#FF6F00", ha="center",
                    fontweight="bold"
                )

        if stuck:
            # "Stuck" label with arrow pointing at barrier
            ax_s.annotate(
                "Stuck!\nCannot cross\nenergy barrier",
                xy=(3.3, 0.20), fontsize=9, color="#B71C1C",
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE",
                          ec="#B71C1C", alpha=0.9)
            )

        ax_s.set_xlim(0, 10)
        ax_s.set_ylim(0, 1.25)
        ax_s.set_xlabel("Search Space Position", fontsize=10)
        ax_s.set_ylabel("Fitness", fontsize=10)
        ax_s.set_title(title, fontsize=11)
        ax_s.annotate("Global\nOptimum", xy=(8.5, 1.10), fontsize=8,
                      ha="center", color="#2E7D32", fontweight="bold")
        sns.despine(ax=ax_s)

    plt.tight_layout()
    fig2.savefig(
        os.path.join(GRAPHS_DIR, "quantum_tunnelling_static.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig2)
    print("  Saved quantum_tunnelling.gif + quantum_tunnelling_static.png")


# ======================================================================
# 9. TRAINING COST COMPARISON
# ======================================================================

# Maps each architecture to:
#   - glob pattern matching its training energy CSV(s) in data/results/
#   - the model name as it appears in inference_benchmark.csv (for accuracy lookup)
#   - display label and colour
_TRAINING_CSV_CONFIGS = [
    (
        "Custom CNN\n(20 ep, scratch)",
        "cifar10_baseline_energy_metrics_*.csv",
        "Baseline (FP32)",
        "#2196F3",
    ),
    (
        "ResNet-18\n(10 ep, fine-tune)",
        "resnet18_baseline_energy_metrics_*.csv",
        "RN18 Baseline",
        "#E53935",
    ),
    (
        "VGG-16\n(10 ep, fine-tune)",
        "vgg16_baseline_energy_metrics_*.csv",
        "VGG16 Baseline",
        "#7B1FA2",
    ),
]


def load_training_data() -> dict:
    """
    Load training cost data from energy tracker CSVs in data/results/.

    For each architecture the most recent matching CSV is used (files are
    named with a timestamp suffix so lexicographic sort gives the newest).
    Accuracy is sourced from inference_benchmark.csv so it stays in sync
    with the rest of the benchmark results.
    """
    results_dir = os.path.join(".", "data", "results")

    # Load accuracy from inference benchmark
    accuracy_map: dict[str, float] = {}
    if os.path.isfile(CSV_PATH):
        try:
            df_bench = pd.read_csv(CSV_PATH)
            accuracy_map = dict(zip(df_bench["model"], df_bench["accuracy_pct"]))
        except Exception as exc:
            print(f"  [WARNING] Could not read {CSV_PATH} for accuracy: {exc}")

    training_data: dict = {}
    for label, pattern, bench_model, colour in _TRAINING_CSV_CONFIGS:
        matches = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if not matches:
            print(f"  [WARNING] No training CSV found matching: {pattern}")
            continue
        # Pick the file with the longest duration — this is always the training run,
        # not a short inference/benchmark run that shares the same naming prefix.
        best_file = None
        best_duration = -1.0
        for path in matches:
            try:
                row = pd.read_csv(path).iloc[0]
                dur = float(row["duration_seconds"])
                if dur > best_duration:
                    best_duration = dur
                    best_file = path
            except Exception:
                pass
        if best_file is None:
            print(f"  [WARNING] Could not read any CSV matching: {pattern}")
            continue
        try:
            row = pd.read_csv(best_file).iloc[0]
            training_data[label] = {
                "duration_s":   float(row["duration_seconds"]),
                "energy_j":     float(row["total_energy_joules"]),
                "cpu_energy_j": float(row["cpu_energy_joules"]),
                "gpu_energy_j": float(row["gpu_energy_joules"]),
                "co2_g":        float(row["emissions_gco2"]),
                "accuracy":     float(accuracy_map.get(bench_model, 0.0)),
                "colour":       colour,
            }
            print(f"  Loaded training CSV: {os.path.basename(best_file)} ({best_duration:.0f}s)")
        except Exception as exc:
            print(f"  [WARNING] Failed to load {best_file}: {exc}")

    return training_data


def plot_training_cost_comparison():
    """
    Three-panel chart comparing training cost across architectures:
      (a) Training duration (s)
      (b) Total energy (J) broken down by CPU vs GPU
      (c) CO₂ emissions (g) — annotated with final accuracy
    Includes a note that ResNet/VGG costs are fine-tuning only.
    """
    print("  Generating training cost comparison...")

    TRAINING_DATA = load_training_data()
    if not TRAINING_DATA:
        print("  [SKIP] No training data loaded — skipping training cost comparison.")
        return

    labels   = list(TRAINING_DATA.keys())
    colours  = [v["colour"] for v in TRAINING_DATA.values()]
    duration = [v["duration_s"]   for v in TRAINING_DATA.values()]
    energy   = [v["energy_j"]     for v in TRAINING_DATA.values()]
    cpu_e    = [v["cpu_energy_j"] for v in TRAINING_DATA.values()]
    gpu_e    = [v["gpu_energy_j"] for v in TRAINING_DATA.values()]
    co2      = [v["co2_g"]        for v in TRAINING_DATA.values()]
    acc      = [v["accuracy"]     for v in TRAINING_DATA.values()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        "Training Cost Comparison — Custom CNN vs Pretrained Models\n"
        "(RTX 3070 Laptop + Ryzen 7 6800H  |  ResNet-18 & VGG-16 are fine-tuning costs only)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x = np.arange(len(labels))

    # ── Panel A: Training duration ──────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, duration, color=colours, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, duration):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Training Duration (s)", fontsize=11)
    ax.set_title("(a) Training Time", fontsize=12)
    ax.set_ylim(0, max(duration) * 1.2)
    sns.despine(ax=ax)

    # ── Panel B: Stacked CPU vs GPU energy ─────────────────────────
    ax = axes[1]
    bars_gpu = ax.bar(x, gpu_e, color=colours, edgecolor="white", linewidth=0.8,
                      width=0.5, label="GPU energy")
    bars_cpu = ax.bar(x, cpu_e, bottom=gpu_e, color=[c + "88" for c in colours],
                      edgecolor="white", linewidth=0.8, width=0.5, label="CPU energy")
    for i, (g, c, e) in enumerate(zip(gpu_e, cpu_e, energy)):
        ax.text(x[i], e + 200, f"{e:,.0f} J", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total Energy (Joules)", fontsize=11)
    ax.set_title("(b) Training Energy (CPU + GPU)", fontsize=12)
    ax.set_ylim(0, max(energy) * 1.22)
    ax.legend(fontsize=9, loc="upper right")
    sns.despine(ax=ax)

    # ── Panel C: CO₂ with accuracy overlay ─────────────────────────
    ax = axes[2]
    bars = ax.bar(x, co2, color=colours, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, c_val, a_val in zip(bars, co2, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{c_val:.4f} g\n({a_val:.2f}% acc)", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CO₂ Emissions (g)", fontsize=11)
    ax.set_title("(c) CO₂ Emissions (with final accuracy)", fontsize=12)
    ax.set_ylim(0, max(co2) * 1.3)
    sns.despine(ax=ax)

    # ── Footnote ────────────────────────────────────────────────────
    fig.text(
        0.5, -0.04,
        "⚠  ResNet-18 and VGG-16 figures reflect CIFAR-10 fine-tuning only (10 epochs).\n"
        "    Original ImageNet pre-training required orders-of-magnitude more compute\n"
        "    (estimated ~1,000× GPU-hours for ResNet-18 on ImageNet — not measured here).",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "training_cost_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved training_cost_comparison.png")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 60)
    print("  GENERATING VISUALISATIONS")
    print("=" * 60)

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(ANIM_DIR, exist_ok=True)

    # Load benchmark data
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run the benchmark first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} models from {CSV_PATH}")

    # Static graphs
    generate_bar_charts(df)
    scatter_accuracy_vs_energy(df)
    scatter_accuracy_vs_throughput(df)
    grouped_category_comparison(df)
    radar_chart(df)
    stacked_energy_chart(df)

    # Architecture comparison charts
    print("\nGenerating architecture comparison charts...")
    architecture_baseline_comparison(df)
    architecture_optimisation_heatmap(df)
    architecture_best_comparison(df)
    architecture_efficiency_scatter(df)

    # Training cost comparison
    print("\nGenerating training cost charts...")
    plot_training_cost_comparison()

    # Animations
    print("\nGenerating algorithm animations...")
    animate_qiga_convergence()
    animate_qisa_cooling()
    animate_pruning_effect()
    animate_structured_pruning()
    animate_quantum_tunnelling()

    print(f"\nAll visualisations saved to {GRAPHS_DIR}/")
    print(f"Animations saved to {ANIM_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
