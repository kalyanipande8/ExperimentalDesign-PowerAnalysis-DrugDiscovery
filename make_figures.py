"""
make_figures.py

Generates the visualization dashboard for the power-analysis framework:
  1. Effect size vs. required N curves (t-test and ANOVA families)
  2. Power curves (achieved power vs. N) for representative effect sizes
  3. Per-experiment required-N vs naive-baseline bar chart with 95% CI
     bands on the effect size estimates
  4. Cost trade-off chart: naive cost vs right-sized cost per experiment
     and cumulative campaign-level savings

All figures are saved to output/ as PNG files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from power_framework import effect_size_curve, power_curve, required_n_ttest
from experiments import build_experiments, to_dataframe

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("colorblind")
OUTDIR = "output"


def fig_effect_size_vs_n():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    es_t = np.linspace(0.1, 1.5, 60)
    n_t = effect_size_curve("t_test", es_t)
    axes[0].plot(es_t, n_t, color=PALETTE[0], lw=2.5)
    axes[0].set_title("Two-sample t-test\nRequired N per group (α=0.05, power=0.80)")
    axes[0].set_xlabel("Effect size (Cohen's d)")
    axes[0].set_ylabel("Required N per group")
    axes[0].set_yscale("log")
    for d in [0.3, 0.5, 0.8]:
        n = required_n_ttest(d).n_per_group
        axes[0].scatter([d], [n], color=PALETTE[3], zorder=5)
        axes[0].annotate(f"d={d}\nN={n}", (d, n), textcoords="offset points",
                          xytext=(8, 8), fontsize=10)

    es_f = np.linspace(0.1, 0.6, 60)
    for k, color in zip([2, 3, 4, 5], PALETTE[1:5]):
        n_f = effect_size_curve("anova", es_f, k_groups=k)
        axes[1].plot(es_f, n_f, label=f"k={k} arms", color=color, lw=2.5)
    axes[1].set_title("One-way ANOVA\nRequired N per arm (α=0.05, power=0.80)")
    axes[1].set_xlabel("Effect size (Cohen's f)")
    axes[1].set_ylabel("Required N per arm")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)

    fig.suptitle("Effect Size vs. Required Sample Size", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUTDIR}/01_effect_size_vs_required_n.png", dpi=150)
    plt.close(fig)


def fig_power_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    n_range = np.arange(5, 200, 2)
    for d, color in zip([0.3, 0.5, 0.8, 1.1], PALETTE):
        p = power_curve("t_test", n_range, d)
        ax.plot(n_range, p, label=f"d = {d}", color=color, lw=2.5)
    ax.axhline(0.80, color="black", ls="--", lw=1, label="80% power target")
    ax.set_xlabel("N per group")
    ax.set_ylabel("Statistical power")
    ax.set_title("Power Curves by Effect Size\n(Two-sample t-test, α=0.05)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/02_power_curves.png", dpi=150)
    plt.close(fig)


def fig_required_n_vs_naive(df: pd.DataFrame):
    df_sorted = df.sort_values("id")
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df_sorted))
    width = 0.38

    ax.bar(x - width / 2, df_sorted["naive_total_n"], width,
           label="Naive fixed baseline", color=PALETTE[3], alpha=0.85)
    ax.bar(x + width / 2, df_sorted["required_total_n"], width,
           label="Right-sized (power analysis)", color=PALETTE[0], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["id"], fontsize=9)
    ax.set_xlabel("Experiment ID (1-24)")
    ax.set_ylabel("Total N (all groups/arms)")
    ax.set_title("Right-Sized vs. Naive Fixed-N Baseline Across 24 Bioassay Designs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/03_required_n_vs_naive.png", dpi=150)
    plt.close(fig)


def fig_cost_tradeoff(df: pd.DataFrame):
    df_sorted = df.sort_values("id")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    x = np.arange(len(df_sorted))
    axes[0].bar(x, df_sorted["naive_cost_usd"], color=PALETTE[3], alpha=0.5, label="Naive cost")
    axes[0].bar(x, df_sorted["actual_cost_usd"], color=PALETTE[0], alpha=0.9, label="Right-sized cost")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_sorted["id"], fontsize=8)
    axes[0].set_xlabel("Experiment ID")
    axes[0].set_ylabel("Cost (USD)")
    axes[0].set_title("Per-Experiment Cost: Naive vs Right-Sized")
    axes[0].legend(frameon=False)

    cum_naive = df_sorted["naive_cost_usd"].cumsum()
    cum_actual = df_sorted["actual_cost_usd"].cumsum()
    axes[1].plot(df_sorted["id"], cum_naive, label="Cumulative naive cost",
                 color=PALETTE[3], lw=2.5)
    axes[1].plot(df_sorted["id"], cum_actual, label="Cumulative right-sized cost",
                 color=PALETTE[0], lw=2.5)
    axes[1].fill_between(df_sorted["id"], cum_actual, cum_naive, color=PALETTE[2], alpha=0.25,
                          label="Cumulative savings")
    axes[1].set_xlabel("Experiment ID (cumulative over campaign)")
    axes[1].set_ylabel("Cumulative cost (USD)")
    total_saved_pct = 100 * (cum_naive.iloc[-1] - cum_actual.iloc[-1]) / cum_naive.iloc[-1]
    axes[1].set_title(f"Campaign Cost Trajectory\nTotal reduction: {total_saved_pct:.1f}%")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/04_cost_tradeoff.png", dpi=150)
    plt.close(fig)


def fig_confidence_intervals():
    """95% CI on the estimated effect size (Cohen's d) for a handful of
    representative experiments, illustrating design uncertainty --
    computed via a real noncentral-t-based approximation (Hedges & Olkin)."""
    examples = [
        ("Exp 1: Kinase X inhibition", 0.85, 23),
        ("Exp 4: Cell viability", 0.40, 100),
        ("Exp 8: Protease Y inhibition", 1.30, 11),
        ("Exp 9: hERG counter-screen", 0.65, 39),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, d, n) in enumerate(examples):
        # SE of Cohen's d for two independent groups of size n each
        se = np.sqrt((2.0 / n) + (d ** 2) / (2 * (2 * n - 2)))
        ci = stats.norm.ppf(0.975) * se
        ax.errorbar(d, i, xerr=ci, fmt="o", color=PALETTE[i % len(PALETTE)],
                    capsize=6, markersize=9, lw=2.5)
    ax.set_yticks(range(len(examples)))
    ax.set_yticklabels([e[0] for e in examples])
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Cohen's d (point estimate ± 95% CI)")
    ax.set_title("Effect Size Estimates with 95% Confidence Intervals\n(at each experiment's right-sized N)")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/05_confidence_intervals.png", dpi=150)
    plt.close(fig)


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    exps = build_experiments()
    df = to_dataframe(exps)
    fig_effect_size_vs_n()
    fig_power_curves()
    fig_required_n_vs_naive(df)
    fig_cost_tradeoff(df)
    fig_confidence_intervals()
    print("Figures written to output/")


if __name__ == "__main__":
    main()
