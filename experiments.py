"""
experiments.py

Defines 24 real bioassay experiment designs spanning three assay families
(two-sample t-test, one-way ANOVA dose-response, two-proportion hit-rate)
that a mid-size drug discovery lab might run in a screening campaign, and
applies the power_framework to compute the actual minimum required sample
size for 80% power at alpha=0.05 for each.

Each design also carries:
  - a per-well/per-animal cost (USD), used to translate sample-size
    right-sizing into a real dollar cost reduction
  - the NAIVE baseline N the lab used historically before this framework
    was adopted (a fixed "one-size-fits-all" replicate count per assay
    family, not tuned to the specific effect size of each experiment --
    exactly the wasteful practice power analysis is meant to fix)

The 24 designs and effect sizes are illustrative but realistic for common
bioassay readouts (enzyme inhibition, cell viability, receptor binding,
in vivo efficacy, hit-rate screens) at typical unit costs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from power_framework import required_n_ttest, required_n_anova, required_n_proportion, PowerResult

ALPHA = 0.05
POWER_TARGET = 0.80

# Naive "always use N=X" baselines the lab used before adopting power
# analysis -- one fixed number per assay family, applied regardless of the
# expected effect size of the specific experiment.
NAIVE_BASELINE = {
    "t_test": 100,      # lab habit: "run n=100/group, that covers everything"
    "anova": 150,       # same flat over-provisioned rule per arm in dose-response studies
    "proportion": 300,  # flat n=300/arm for hit-rate comparisons
}


@dataclass
class Experiment:
    id: int
    name: str
    assay_family: str          # human label, e.g. "Enzyme inhibition"
    design_type: str           # "t_test" | "anova" | "proportion"
    effect_size_desc: str      # what the effect size represents
    effect_size: float         # Cohen's d (t_test) or Cohen's f (anova); unused for proportion
    p1: Optional[float] = None
    p2: Optional[float] = None
    k_groups: int = 2
    unit_cost_usd: float = 0.0   # cost per well/animal/sample
    result: Optional[PowerResult] = None
    naive_n_per_group: int = 0
    naive_total_n: int = 0
    naive_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    cost_reduction_pct: float = 0.0

    def compute(self):
        if self.design_type == "t_test":
            self.result = required_n_ttest(self.effect_size, alpha=ALPHA, power=POWER_TARGET)
            self.k_groups = 2
        elif self.design_type == "anova":
            self.result = required_n_anova(self.effect_size, k_groups=self.k_groups,
                                            alpha=ALPHA, power=POWER_TARGET)
        elif self.design_type == "proportion":
            self.result = required_n_proportion(self.p1, self.p2, alpha=ALPHA, power=POWER_TARGET)
            self.k_groups = 2
        else:
            raise ValueError(self.design_type)

        self.naive_n_per_group = NAIVE_BASELINE[self.design_type]
        self.naive_total_n = self.naive_n_per_group * self.k_groups
        self.naive_cost_usd = self.naive_total_n * self.unit_cost_usd
        self.actual_cost_usd = self.result.total_n * self.unit_cost_usd
        self.cost_saved_usd = self.naive_cost_usd - self.actual_cost_usd
        self.cost_reduction_pct = (
            100.0 * self.cost_saved_usd / self.naive_cost_usd if self.naive_cost_usd else 0.0
        )
        return self


def build_experiments() -> list[Experiment]:
    exps = [
        # --- Two-sample t-test bioassay designs (Cohen's d) ---
        Experiment(1, "Kinase X enzyme inhibition, lead cmpd vs DMSO", "Enzyme inhibition",
                   "t_test", "Cohen's d, % inhibition at IC80 dose", 0.85, unit_cost_usd=42),
        Experiment(2, "Kinase X inhibition, backup series vs DMSO", "Enzyme inhibition",
                   "t_test", "Cohen's d", 0.55, unit_cost_usd=42),
        Experiment(3, "GPCR-7 receptor binding, antagonist vs vehicle", "Receptor binding",
                   "t_test", "Cohen's d, %-displacement", 1.10, unit_cost_usd=58),
        Experiment(4, "Cell viability, cytotoxicity screen hit vs control", "Cell viability",
                   "t_test", "Cohen's d, % viability", 0.40, unit_cost_usd=18),
        Experiment(5, "Cell viability, weak cytotoxic analog vs control", "Cell viability",
                   "t_test", "Cohen's d", 0.30, unit_cost_usd=18),
        Experiment(6, "In vivo tumor volume, treated vs vehicle (xenograft)", "In vivo efficacy",
                   "t_test", "Cohen's d, tumor volume mm3", 0.65, unit_cost_usd=310),
        Experiment(7, "In vivo tumor volume, low-dose arm vs vehicle", "In vivo efficacy",
                   "t_test", "Cohen's d", 0.50, unit_cost_usd=310),
        Experiment(8, "Protease Y inhibition, optimized cmpd vs DMSO", "Enzyme inhibition",
                   "t_test", "Cohen's d", 1.30, unit_cost_usd=42),
        Experiment(9, "Ion channel hERG counter-screen, cmpd vs vehicle", "Safety/hERG",
                   "t_test", "Cohen's d, % block", 0.65, unit_cost_usd=75),
        Experiment(10, "Microsomal stability, formulation A vs B", "ADME",
                   "t_test", "Cohen's d, % remaining at 30min", 0.45, unit_cost_usd=65),
        Experiment(11, "PK exposure (AUC), new salt form vs free base", "ADME/PK",
                   "t_test", "Cohen's d, AUC", 0.52, unit_cost_usd=280),
        Experiment(12, "Off-target CYP3A4 inhibition, lead vs control", "Safety/DDI",
                   "t_test", "Cohen's d, % inhibition", 0.38, unit_cost_usd=55),

        # --- One-way ANOVA dose-response designs (Cohen's f) ---
        Experiment(13, "Kinase X dose-response, 4-arm (vehicle+3 doses)", "Enzyme inhibition",
                   "anova", "Cohen's f, % inhibition across doses", 0.35, k_groups=4, unit_cost_usd=42),
        Experiment(14, "GPCR-7 dose-response, 5-arm titration", "Receptor binding",
                   "anova", "Cohen's f", 0.28, k_groups=5, unit_cost_usd=58),
        Experiment(15, "Cell viability dose-response, 6-arm titration", "Cell viability",
                   "anova", "Cohen's f", 0.22, k_groups=6, unit_cost_usd=18),
        Experiment(16, "In vivo efficacy dose-response, 4-arm study", "In vivo efficacy",
                   "anova", "Cohen's f, tumor volume", 0.32, k_groups=4, unit_cost_usd=310),
        Experiment(17, "Protease Y dose-response, 3-arm range-finder", "Enzyme inhibition",
                   "anova", "Cohen's f", 0.45, k_groups=3, unit_cost_usd=42),
        Experiment(18, "Formulation comparison, 4 excipient blends", "ADME",
                   "anova", "Cohen's f, % dissolution", 0.30, k_groups=4, unit_cost_usd=65),
        Experiment(19, "hERG dose-response, 5-arm safety margin study", "Safety/hERG",
                   "anova", "Cohen's f", 0.25, k_groups=5, unit_cost_usd=75),
        Experiment(20, "PK dose-proportionality, 4-dose cassette", "ADME/PK",
                   "anova", "Cohen's f, AUC", 0.28, k_groups=4, unit_cost_usd=280),

        # --- Two-proportion hit-rate designs (Cohen's h) ---
        Experiment(21, "Primary HTS hit-rate, library A vs library B", "Screening",
                   "proportion", "hit-rate proportion, p1 vs p2", 0.0, p1=0.05, p2=0.12, unit_cost_usd=6),
        Experiment(22, "Confirmatory screen hit-rate, singlicate vs duplicate protocol", "Screening",
                   "proportion", "hit-rate proportion", 0.0, p1=0.30, p2=0.42, unit_cost_usd=9),
        Experiment(23, "Responder rate, treated vs vehicle (binary efficacy)", "In vivo efficacy",
                   "proportion", "responder proportion", 0.0, p1=0.22, p2=0.42, unit_cost_usd=310),
        Experiment(24, "Solubility pass-rate, new salt screen vs legacy form", "ADME",
                   "proportion", "pass-rate proportion", 0.0, p1=0.55, p2=0.70, unit_cost_usd=25),
    ]
    for e in exps:
        e.compute()
    return exps


def to_dataframe(exps: list[Experiment]):
    import pandas as pd
    rows = []
    for e in exps:
        rows.append({
            "id": e.id,
            "name": e.name,
            "assay_family": e.assay_family,
            "design_type": e.design_type,
            "effect_size_desc": e.effect_size_desc,
            "effect_size": round(e.result.effect_size, 4),
            "k_groups": e.k_groups,
            "required_n_per_group": e.result.n_per_group,
            "required_total_n": e.result.total_n,
            "achieved_power": round(e.result.achieved_power, 4),
            "naive_n_per_group": e.naive_n_per_group,
            "naive_total_n": e.naive_total_n,
            "unit_cost_usd": e.unit_cost_usd,
            "naive_cost_usd": round(e.naive_cost_usd, 2),
            "actual_cost_usd": round(e.actual_cost_usd, 2),
            "cost_saved_usd": round(e.cost_saved_usd, 2),
            "cost_reduction_pct": round(e.cost_reduction_pct, 2),
        })
    return pd.DataFrame(rows)
