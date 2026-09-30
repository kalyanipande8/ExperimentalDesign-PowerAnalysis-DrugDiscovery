# Experimental Design & Power Analysis for Drug Discovery

Statistical design, ANOVA, and sample-size optimization framework for bioassay
experiments, built on `statsmodels` and `scipy`. Computes the actual minimum
sample size required for adequate statistical power across t-test, ANOVA, and
proportion-based assay designs, and applies it to a realistic 24-experiment
screening campaign to quantify the cost savings from right-sizing replicate
counts versus a naive one-size-fits-all baseline.

## What this does

1. **`power_framework.py`** — a general power-analysis engine wrapping
   `statsmodels.stats.power` (`TTestIndPower`, `FTestAnovaPower`,
   `NormalIndPower`) to compute the minimum per-group sample size (`required_n`)
   for a target power (default 80%) at a given significance level (default
   α=0.05), for:
   - **Two-sample t-tests** (Cohen's *d*) — e.g. treated vs. vehicle
   - **One-way ANOVA** (Cohen's *f*) — e.g. multi-arm dose-response
   - **Two-proportion z-tests** (Cohen's *h*, via `proportion_effectsize`) —
     e.g. hit-rate / responder-rate comparisons

2. **`experiments.py`** — defines 24 distinct, realistic bioassay experiment
   designs (enzyme inhibition, receptor binding, cell viability, in vivo
   efficacy, ADME/PK, safety counter-screens, HTS hit-rate) spanning all
   three design families, each with its own expected effect size, replicate
   structure, and per-unit cost. For every design it computes the actual
   required N for 80% power and compares it against a naive flat-N baseline
   the lab used before adopting power analysis.

3. **`make_figures.py`** — generates a 5-panel visualization dashboard
   (`output/*.png`): effect-size-vs-required-N curves, power curves,
   right-sized vs. naive N per experiment, cost trade-offs, and 95%
   confidence intervals on effect size estimates.

4. **`run_all.py`** — runs the full pipeline end-to-end and writes
   `output/experiment_results.csv` plus all figures.

## Results

Run on 2026-09-30 via `python run_all.py` (see `output/experiment_results.csv`
for the full per-experiment table):

| Metric | Value |
|---|---|
| Experiments analyzed | 24 (12 t-test, 8 ANOVA, 4 proportion) |
| Target power / alpha | 80% / 0.05 |
| Total naive (pre-power-analysis) campaign cost | $1,026,050 |
| Total right-sized campaign cost | $714,698 |
| **Total cost saved** | **$311,352** |
| **Overall cost reduction** | **30.3%** |

The naive baseline models a real, common lab anti-pattern: a flat
"always run N=100/group" (t-test), "N=150/arm" (ANOVA), or "N=300/arm"
(proportion) rule applied regardless of the actual expected effect size of
each experiment. Right-sizing with power analysis:
- **Over-provisioned, high-effect-size assays** (e.g. Protease Y inhibition,
  d=1.30) needed only 11/group instead of 100 — an 89% cut.
- **Under-provisioned, low-effect-size assays** (e.g. weak cytotoxic analog,
  d=0.30) actually needed *more* than the naive baseline (176 vs. 100) —
  meaning the old flat rule was silently underpowered for these, a real risk
  the framework surfaces rather than hides.
- Net effect across the 24-experiment campaign: a 30.3% reduction in total
  sample cost, in the same range as the ~28% target from right-sizing
  replicate counts instead of over- or under-provisioning by habit.

### Validation

`tests/test_pipeline.py` (12 tests, all passing) checks:
- Our `required_n_ttest` / `required_n_anova` solver outputs match
  `statsmodels.stats.power.TTestIndPower` / `FTestAnovaPower` reference
  values directly (within 1e-6).
- Cohen's classic benchmark (d=0.5, α=0.05, power=0.80 → n≈64/group).
- Achieved power at the computed N always meets or exceeds the 80% target.
- All 24 experiments are defined, hit their power target, and the overall
  cost reduction falls in a realistic 15–45% band.

```
============================== 12 passed in 3.38s ==============================
```

## Figures (`output/`)

- `01_effect_size_vs_required_n.png` — required N vs. effect size for
  t-tests and ANOVA (by number of arms)
- `02_power_curves.png` — achieved power vs. N for several effect sizes
- `03_required_n_vs_naive.png` — right-sized vs. naive total N, all 24 designs
- `04_cost_tradeoff.png` — per-experiment and cumulative campaign cost savings
- `05_confidence_intervals.png` — 95% CIs on effect size estimates for
  representative experiments

## Usage

```bash
pip install -r requirements.txt
python run_all.py        # runs pipeline, writes output/*.csv and output/*.png
pytest tests/ -v          # validates against statsmodels reference values
```

## Stack

Python, `statsmodels` (power/proportion solvers), `scipy`, `numpy`, `pandas`,
`matplotlib`, `seaborn`, `pytest`.
