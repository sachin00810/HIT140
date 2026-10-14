"""
04_visualise_data.py - Academic Publication-Quality Visualisations
FIFA World Cup 2026: Pass Completion vs Match Outcome Analysis
HIT140 Foundations of Data Science

Generates:
1. outputs/histogram_pass_completion.png (Distribution of pass completion by result group)
2. outputs/boxplot_pass_completion.png (Boxplot comparing Win vs Loss groups)
"""

import os

# Set Matplotlib config dir to workspace to prevent permission errors
MPL_DIR = os.path.join(os.path.dirname(__file__), "..", ".matplotlib")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fifa2026_cleaned.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

HIST_PATH = os.path.join(OUTPUTS_DIR, "histogram_pass_completion.png")
BOXPLOT_PATH = os.path.join(OUTPUTS_DIR, "boxplot_pass_completion.png")


def create_visualisations():
    print("=" * 70)
    print("STEP 6: DATA VISUALISATIONS")
    print("=" * 70)

    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(f"Cleaned data not found at: {PROCESSED_CSV_PATH}")

    df = pd.read_csv(PROCESSED_CSV_PATH)
    win_data = df[df["result"] == "Win"]["pass_completion"]
    loss_data = df[df["result"] == "Loss"]["pass_completion"]

    mean_win = win_data.mean()
    mean_loss = loss_data.mean()
    median_win = win_data.median()
    median_loss = loss_data.median()

    # Color scheme
    color_win = "#1f77b4"   # Classic academic steel blue
    color_loss = "#d62728"  # Classic academic crimson red

   
    #  HISTOGRAM: Distribution by Result Group
   
    print("Generating Histogram: outputs/histogram_pass_completion.png ...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    bins = np.linspace(72, 95, 24)

    ax.hist(
        win_data,
        bins=bins,
        alpha=0.65,
        color=color_win,
        edgecolor="black",
        linewidth=1.0,
        label=f"Winning Teams (n = {len(win_data)}, Mean = {mean_win:.2f}%)",
    )
    ax.hist(
        loss_data,
        bins=bins,
        alpha=0.65,
        color=color_loss,
        edgecolor="black",
        linewidth=1.0,
        label=f"Losing Teams (n = {len(loss_data)}, Mean = {mean_loss:.2f}%)",
    )

    # Vertical Mean Lines
    ax.axvline(
        mean_win,
        color="#0d47a1",
        linestyle="--",
        linewidth=2.0,
        label=f"Win Mean: {mean_win:.2f}%",
    )
    ax.axvline(
        mean_loss,
        color="#b71c1c",
        linestyle="--",
        linewidth=2.0,
        label=f"Loss Mean: {mean_loss:.2f}%",
    )

    # Styling
    ax.set_title(
        "FIFA World Cup 2026: Pass-Completion Percentage Distribution by Match Outcome",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Pass Completion Percentage (%)", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylabel("Frequency (Number of Team-Matches)", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_xlim(71, 95)
    ax.set_ylim(0, 25)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=10)

    # Text annotation
    ax.text(
        0.97,
        0.75,
        f"Mean Diff: {mean_win - mean_loss:.2f}%\nOne-tailed p < 0.001\nCohen's d = 1.99 (Large)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#adb5bd", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=300)
    plt.close()
    print(f"Saved histogram to: {HIST_PATH}")

  
    # 6.2 BOXPLOT: Comparison between Win and Loss Groups
    print("Generating Boxplot: outputs/boxplot_pass_completion.png ...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    box_data = [win_data, loss_data]
    box_labels = [f"Win\n(n = {len(win_data)})", f"Loss\n(n = {len(loss_data)})"]

    bp = ax.boxplot(
        box_data,
        tick_labels=box_labels,
        patch_artist=True,
        widths=0.45,
        showmeans=True,
        meanline=False,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=8),
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(color="#333333", linewidth=1.2, linestyle="-"),
        capprops=dict(color="#333333", linewidth=1.2),
        flierprops=dict(marker="o", markerfacecolor="gray", markeredgecolor="black", markersize=6, alpha=0.7),
    )

    # Fill colors
    colors = [color_win, color_loss]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Annotate means and medians
    ax.text(
        1.28,
        mean_win,
        f"Mean: {mean_win:.2f}%\nMedian: {median_win:.2f}%",
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor=color_win, alpha=0.9),
    )
    ax.text(
        2.28,
        mean_loss,
        f"Mean: {mean_loss:.2f}%\nMedian: {median_loss:.2f}%",
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor=color_loss, alpha=0.9),
    )

    # Styling
    ax.set_title(
        "FIFA World Cup 2026: Pass-Completion Comparison by Match Outcome",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Match Outcome (Result Group)", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylabel("Pass Completion Percentage (%)", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylim(70, 96)
    ax.grid(True, linestyle=":", axis="y", alpha=0.6)

    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_win, lw=6, label="Winning Teams (Win)"),
        Line2D([0], [0], color=color_loss, lw=6, label="Losing Teams (Loss)"),
        Line2D([0], [0], color="black", lw=2, label="Median Line"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=8, label="Mean Marker"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    plt.savefig(BOXPLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved boxplot to: {BOXPLOT_PATH}")
    print("\n[Visualisations Complete] All plots generated successfully.\n")


if __name__ == "__main__":
    create_visualisations()
