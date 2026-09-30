"""
power_framework.py

A general-purpose statistical power analysis framework for bioassay /
drug-discovery experimental design.

Supports three canonical design families used routinely in early-stage
drug discovery bioassays:

  1. Two-sample t-test designs   (e.g. treated vs. vehicle control assay)
  2. One-way ANOVA designs       (e.g. dose-response screen, k arms)
  3. Two-proportion z-test designs (e.g. hit/no-hit binary assay readouts)

For each design family this module wraps the relevant statsmodels power
solver (falling back to closed-form / scipy-based formulas where
statsmodels does not expose a direct solver) and exposes a uniform
`required_n(...)` interface that returns the minimum per-group sample
size needed to achieve a target statistical power at a given alpha and
effect size.

References for validation (see tests/test_pipeline.py):
  - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
  - statsmodels.stats.power.TTestIndPower / FTestAnovaPower / NormalIndPower
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

DesignType = Literal["t_test", "anova", "proportion"]

# Single shared solver instances (statsmodels recommends reuse; stateless).
_TTEST_POWER = TTestIndPower()
_ANOVA_POWER = FTestAnovaPower()
_PROP_POWER = NormalIndPower()


@dataclass
class PowerResult:
    """Result of a power-analysis calculation for one experiment design."""

    design_type: DesignType
    effect_size: float
    alpha: float
    power_target: float
    n_per_group: int          # ceil'd, usable sample size
    n_per_group_exact: float  # unrounded solver output
    groups: int = 2
    achieved_power: float = 0.0
    notes: str = ""

    @property
    def total_n(self) -> int:
        return self.n_per_group * self.groups


def _ceil_n(x: float) -> int:
    """Sample sizes must be whole animals/wells; always round UP so the
    achieved power never falls below the target."""
    return int(math.ceil(round(x, 6)))


def required_n_ttest(effect_size: float, alpha: float = 0.05, power: float = 0.80,
                      ratio: float = 1.0, alternative: str = "two-sided") -> PowerResult:
    """Required per-group N for an independent two-sample t-test, given
    Cohen's d effect size. Wraps statsmodels TTestIndPower.solve_power."""
    n_exact = _TTEST_POWER.solve_power(
        effect_size=effect_size, alpha=alpha, power=power,
        ratio=ratio, alternative=alternative,
    )
    n = _ceil_n(n_exact)
    achieved = _TTEST_POWER.power(
        effect_size=effect_size, nobs1=n, alpha=alpha, ratio=ratio,
        alternative=alternative,
    )
    return PowerResult(
        design_type="t_test", effect_size=effect_size, alpha=alpha,
        power_target=power, n_per_group=n, n_per_group_exact=n_exact,
        groups=2, achieved_power=achieved,
        notes="Two-sample independent t-test (Cohen's d)",
    )


def required_n_anova(effect_size: float, k_groups: int, alpha: float = 0.05,
                      power: float = 0.80) -> PowerResult:
    """Required per-group N for a balanced one-way ANOVA, given Cohen's f
    effect size and number of arms (k_groups). Wraps statsmodels
    FTestAnovaPower.solve_power."""
    n_exact = _ANOVA_POWER.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups,
    )
    n = _ceil_n(n_exact)
    achieved = _ANOVA_POWER.power(
        effect_size=effect_size, nobs=n, alpha=alpha, k_groups=k_groups,
    )
    return PowerResult(
        design_type="anova", effect_size=effect_size, alpha=alpha,
        power_target=power, n_per_group=n, n_per_group_exact=n_exact,
        groups=k_groups, achieved_power=achieved,
        notes=f"One-way ANOVA, k={k_groups} arms (Cohen's f)",
    )


def required_n_proportion(p1: float, p2: float, alpha: float = 0.05,
                           power: float = 0.80, ratio: float = 1.0,
                           alternative: str = "two-sided") -> PowerResult:
    """Required per-group N for a two-proportion z-test (e.g. hit-rate
    comparison between two bioassay conditions). Effect size computed via
    statsmodels' arcsine-transformed proportion_effectsize (Cohen's h)."""
    h = proportion_effectsize(p1, p2)
    n_exact = _PROP_POWER.solve_power(
        effect_size=h, alpha=alpha, power=power, ratio=ratio,
        alternative=alternative,
    )
    n = _ceil_n(n_exact)
    achieved = _PROP_POWER.power(
        effect_size=h, nobs1=n, alpha=alpha, ratio=ratio,
        alternative=alternative,
    )
    return PowerResult(
        design_type="proportion", effect_size=h, alpha=alpha,
        power_target=power, n_per_group=n, n_per_group_exact=n_exact,
        groups=2, achieved_power=achieved,
        notes=f"Two-proportion z-test, p1={p1}, p2={p2} (Cohen's h={h:.3f})",
    )


def effect_size_curve(design_type: DesignType, effect_sizes, alpha: float = 0.05,
                       power: float = 0.80, k_groups: int = 3):
    """Compute required N across a range of effect sizes, for plotting
    effect-size-vs-required-N curves."""
    ns = []
    for es in effect_sizes:
        if design_type == "t_test":
            r = required_n_ttest(es, alpha=alpha, power=power)
        elif design_type == "anova":
            r = required_n_anova(es, k_groups=k_groups, alpha=alpha, power=power)
        else:
            raise ValueError(f"Unsupported design_type for curve: {design_type}")
        ns.append(r.n_per_group_exact)
    return np.array(ns)


def power_curve(design_type: DesignType, n_range, effect_size: float,
                 alpha: float = 0.05, k_groups: int = 3):
    """Compute achieved power across a range of per-group N, for a fixed
    effect size — used for power-curve / sensitivity plots."""
    powers = []
    for n in n_range:
        if design_type == "t_test":
            p = _TTEST_POWER.power(effect_size=effect_size, nobs1=n, alpha=alpha)
        elif design_type == "anova":
            p = _ANOVA_POWER.power(effect_size=effect_size, nobs=n, alpha=alpha,
                                    k_groups=k_groups)
        else:
            raise ValueError(f"Unsupported design_type for curve: {design_type}")
        powers.append(p)
    return np.array(powers)
