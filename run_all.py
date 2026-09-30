"""
run_all.py

Entry point: runs the power-analysis framework across all 24 bioassay
experiment designs, writes the results table to output/experiment_results.csv,
prints a campaign-level cost summary, and generates the visualization
dashboard (output/*.png).
"""
from __future__ import annotations

import os

from experiments import build_experiments, to_dataframe
import make_figures


def main():
    os.makedirs("output", exist_ok=True)

    exps = build_experiments()
    df = to_dataframe(exps)
    df.to_csv("output/experiment_results.csv", index=False)

    total_naive_cost = df["naive_cost_usd"].sum()
    total_actual_cost = df["actual_cost_usd"].sum()
    total_saved = total_naive_cost - total_actual_cost
    overall_reduction_pct = 100 * total_saved / total_naive_cost

    print("=" * 70)
    print("EXPERIMENTAL DESIGN & POWER ANALYSIS -- 24 bioassay experiments")
    print("=" * 70)
    print(df[["id", "name", "design_type", "required_n_per_group",
              "naive_n_per_group", "cost_reduction_pct"]].to_string(index=False))
    print("-" * 70)
    print(f"Total naive (pre-power-analysis) campaign cost:  ${total_naive_cost:,.0f}")
    print(f"Total right-sized campaign cost:                 ${total_actual_cost:,.0f}")
    print(f"Total cost saved:                                ${total_saved:,.0f}")
    print(f"Overall cost reduction:                          {overall_reduction_pct:.1f}%")
    print("=" * 70)

    make_figures.main()

    print("\nResults written to output/experiment_results.csv")
    print("Figures written to output/*.png")


if __name__ == "__main__":
    main()
