# Companion code: Model-Assisted Annotation — From Labels to Reliable Estimates

Simulation and diagnostic code for the post. Everything here runs on a laptop in a few minutes and needs only NumPy and Matplotlib.

Tested with Python 3.11, NumPy 2.4, Matplotlib 3.10.

```
pip install numpy matplotlib
cd posts/model-assisted-annotation/_code
python simulate.py            # ~45 s, writes results.json
python render_figures.py      # writes fig_coverage_vs_bias.{svg,png} and fig_routed_sampling.{svg,png}
python ppipp_diagnostic.py    # ~2 min, writes ppipp_diagnostic_results.json
```

## Files

| File | What it is |
|---|---|
| `simulate.py` | Experiments A and B. Seed 20260920. 2,000 replications. |
| `results.json` | Saved output of `simulate.py`; the numbers and figures in the post come from this file. |
| `render_figures.py` | Draws Figures 2 and 3 from `results.json`. |
| `ppipp_diagnostic.py` | Diagnostic for the small negative mean error of PPI++. Seed 20260921. 20,000 replications per setting. |
| `ppipp_diagnostic_results.json`, `ppipp_diagnostic_output.txt` | Saved output of the diagnostic. |

## Common settings

Target: the population prevalence θ = P(Y = 1), not the realized mean of a finite pool. Intervals: normal approximation, estimate ± 1.96 × standard error. N = 30,000 model-labeled items, human budget 300.

## Experiment A: fixed accuracy, varying error asymmetry

Data-generating process: Y ~ Bernoulli(0.20). The model label f is wrong with total probability 0.12; a share s of that error mass is false positives and 1 − s is false negatives, so FPR = 0.12·s/0.80 and FNR = 0.12·(1 − s)/0.20. The sweep uses s on a coarse grid 0, 0.1, …, 1 plus a fine grid 0.44–0.56.

Sampling: in each replication the n = 300 labeled items and the N = 30,000 prediction-only items are drawn **independently** from this process.

Estimators and standard errors:

- Human-only: mean of Y; sd(Y)/√n.
- Model-only (plug-in): mean of f over all n + N items; sd(f)/√(n + N).
- PPI: mean_N(f) + mean_n(Y − f); √(Var(f)/N + Var(Y − f)/n).
- PPI++: mean_n(Y) + λ̂(mean_N(f) − mean_n(f)), with λ̂ = Cov_n(Y, f) / ((1 + n/N)·Var(f)), Var(f) computed on all n + N items, λ̂ clipped to [0, 1]; √(Var(Y − λ̂f)/n + λ̂²·Var(f)/N).

## Experiment B: human labels selected by uncertainty

Data-generating process: Z ~ N(0, 1); Y ~ Bernoulli(σ(3(Z − 1.05))); model score g = σ(3(Z − 0.80)); model label f = 1[g > 0.5]; uncertainty u = 1 − |2g − 1|. The shifted threshold makes the model over-predict the positive class. Population values (from 4 million draws): prevalence 18.3%, accuracy 88.2%, model-only bias +2.9 pp.

Sampling: one pool of N = 30,000 per replication. Each item is reviewed independently (Poisson sampling) with probability π_i = clip(c·u_i, 0.002, 1), where c is a constant fixed from the population so that the expected number of reviews is 300. π_i depends only on the model's score, so it is known for every item.

Estimators:

- Human-only on the routed sample: mean of Y over reviewed items, treated as a simple random sample.
- PPI on the routed sample, treated as random: mean_N(f) + mean_reviewed(Y − f).
- IPW-corrected: mean over the pool of Ỹ_i = f_i + (R_i/π_i)(Y_i − f_i); standard error sd(Ỹ)/√N.
- Uniform reference: the same pseudo-outcome estimator with π_i = 300/30,000 for every item (independent Bernoulli review).

## PPI++ diagnostic

Experiment A design at three error splits (s = 0, 0.5, 1). Four ways of setting λ: oracle (from population moments), estimated on the same 300 labels (unclipped), estimated and clipped to [0, 1], and two-fold cross-fitted (λ estimated on one half, applied to the other, the two half-estimates averaged).

Result: the mean error is −0.07 to −0.23 pp when λ is estimated on the same labels, and between 0.00 and +0.02 pp with an oracle or cross-fitted λ (Monte Carlo standard error about 0.013 pp). Clipping does not explain it. Coverage is 93.8–94.8% for all variants.

## Limits

One estimand (a mean), binary labels, no distribution shift, one routing rule and one floor, and human labels treated as ground truth.
