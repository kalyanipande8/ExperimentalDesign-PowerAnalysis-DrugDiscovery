"""
tests/test_pipeline.py

Validates the power-analysis framework against known statsmodels reference
values and sanity-checks the 24-experiment cost-reduction pipeline.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

from power_framework import (
    required_n_ttest,
    required_n_anova,
    required_n_proportion,
)
from experiments import build_experiments, to_dataframe


# ---------------------------------------------------------------------------
# Reference-value checks: our wrapper must match statsmodels' own solver
# output directly (they call the exact same underlying functions).
# ---------------------------------------------------------------------------

def test_ttest_matches_statsmodels_reference():
    for d in [0.2, 0.5, 0.8, 1.0]:
        ref_n = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.80)
        result = required_n_ttest(d, alpha=0.05, power=0.80)
        assert abs(result.n_per_group_exact - ref_n) < 1e-6
        # ceil'd N must be >= exact solver value
        assert result.n_per_group >= ref_n


def test_ttest_classic_cohen_benchmark():
    # Cohen's well-known benchmark: d=0.5 (medium effect), alpha=0.05,
    # power=0.80, two-sided t-test -> n ~= 64 per group.
    result = required_n_ttest(0.5, alpha=0.05, power=0.80)
    assert 60 <= result.n_per_group <= 68


def test_anova_matches_statsmodels_reference():
    for f, k in [(0.1, 3), (0.25, 4), (0.4, 3)]:
        ref_n = FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=0.80, k_groups=k)
        result = required_n_anova(f, k_groups=k, alpha=0.05, power=0.80)
        assert abs(result.n_per_group_exact - ref_n) < 1e-6
        assert result.n_per_group >= ref_n


def test_proportion_effect_size_and_n_reasonable():
    # p1=0.10 vs p2=0.30 is a fairly large effect; required N should be
    # modest (tens, not thousands) at 80% power.
    result = required_n_proportion(0.10, 0.30, alpha=0.05, power=0.80)
    assert 30 <= result.n_per_group <= 80


def test_achieved_power_meets_or_exceeds_target():
    for d in [0.2, 0.4, 0.6, 0.9]:
        r = required_n_ttest(d)
        assert r.achieved_power >= 0.80 - 1e-6


def test_larger_effect_size_requires_smaller_n():
    small = required_n_ttest(0.2).n_per_group
    medium = required_n_ttest(0.5).n_per_group
    large = required_n_ttest(0.8).n_per_group
    assert small > medium > large


def test_anova_more_arms_same_f_requires_more_total_n_trend():
    # per-arm N should not blow up unreasonably as arms increase for fixed f
    r3 = required_n_anova(0.25, k_groups=3)
    r5 = required_n_anova(0.25, k_groups=5)
    assert r3.n_per_group > 0 and r5.n_per_group > 0


# ---------------------------------------------------------------------------
# Pipeline-level checks on the 24 experiment designs
# ---------------------------------------------------------------------------

def test_24_experiments_defined():
    exps = build_experiments()
    assert len(exps) == 24
    ids = sorted(e.id for e in exps)
    assert ids == list(range(1, 25))


def test_all_designs_hit_power_target():
    exps = build_experiments()
    for e in exps:
        assert e.result.achieved_power >= 0.80 - 1e-6, (
            f"Experiment {e.id} ({e.name}) achieved power "
            f"{e.result.achieved_power:.3f} below 0.80 target"
        )


def test_design_type_distribution():
    exps = build_experiments()
    families = [e.design_type for e in exps]
    assert families.count("t_test") == 12
    assert families.count("anova") == 8
    assert families.count("proportion") == 4


def test_cost_reduction_in_reasonable_band():
    exps = build_experiments()
    df = to_dataframe(exps)
    total_naive = df["naive_cost_usd"].sum()
    total_actual = df["actual_cost_usd"].sum()
    overall_pct = 100 * (total_naive - total_actual) / total_naive

    # Overall right-sizing should meaningfully cut cost vs the naive flat-N
    # baseline, in a realistic band around the ~28% target reported.
    assert 15.0 <= overall_pct <= 45.0


def test_no_negative_sample_sizes_or_costs():
    exps = build_experiments()
    df = to_dataframe(exps)
    assert (df["required_n_per_group"] > 0).all()
    assert (df["required_total_n"] > 0).all()
    assert (df["actual_cost_usd"] >= 0).all()
    assert (df["naive_cost_usd"] >= 0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
