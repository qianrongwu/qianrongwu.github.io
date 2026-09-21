"""Coverage simulation for "Labels as a Control System", section 5 (Correct).

Experiment A: fixed 88% surrogate accuracy, vary how the 12% of errors split
              between false positives and false negatives. Random human sample.
              Estimators: human-only, plug-in, PPI, PPI++.
Experiment B: human labels are routed by model uncertainty (non-random).
              Estimators: naive human-only, naive PPI, IPW-corrected (DSL-style),
              plus a uniform-sampling reference at the same expected budget.

Target in both: prevalence theta = P(Y = 1). Nominal 95% Wald intervals.
"""
import json
import numpy as np

Z = 1.959964
REPS = 2000
N_POOL = 30_000
N_HUMAN = 300
rng = np.random.default_rng(20260920)


def summarize(est, se, theta):
    est, se = np.asarray(est), np.asarray(se)
    lo, hi = est - Z * se, est + Z * se
    return {
        "bias": float(est.mean() - theta),
        "rmse": float(np.sqrt(((est - theta) ** 2).mean())),
        "coverage": float(((lo <= theta) & (theta <= hi)).mean()),
        "ci_width": float((hi - lo).mean()),
    }


# ---------------------------------------------------------------- Experiment A
def draw_confusion(n, prev, fpr, fnr):
    y = rng.random(n) < prev
    u = rng.random(n)
    f = np.where(y, u >= fnr, u < fpr)
    return y.astype(float), f.astype(float)


def experiment_a(prev=0.20, err=0.12, fp_shares=None):
    if fp_shares is None:   # coarse grid + fine grid around the point where errors cancel
        fp_shares = np.unique(np.round(np.concatenate(
            [np.linspace(0, 1, 11), np.arange(0.44, 0.561, 0.01)]), 2))
    out = []
    for share in fp_shares:
        p_fp, p_fn = err * share, err * (1 - share)      # joint error masses
        fpr, fnr = p_fp / (1 - prev), p_fn / prev
        res = {k: ([], []) for k in ("human_only", "plug_in", "ppi", "ppi_pp")}
        for _ in range(REPS):
            y, f = draw_confusion(N_HUMAN, prev, fpr, fnr)   # human-labeled
            _, fu = draw_confusion(N_POOL, prev, fpr, fnr)   # model-labeled only
            n, N = N_HUMAN, N_POOL
            # human-only
            res["human_only"][0].append(y.mean())
            res["human_only"][1].append(y.std(ddof=1) / np.sqrt(n))
            # plug-in: treat model labels as outcomes
            allf = np.concatenate([f, fu])
            res["plug_in"][0].append(allf.mean())
            res["plug_in"][1].append(allf.std(ddof=1) / np.sqrt(n + N))
            # PPI: model mean + rectifier
            rect = y - f
            res["ppi"][0].append(fu.mean() + rect.mean())
            res["ppi"][1].append(np.sqrt(fu.var(ddof=1) / N + rect.var(ddof=1) / n))
            # PPI++: power-tuned lambda
            var_f = allf.var(ddof=1)
            lam = 0.0 if var_f == 0 else np.cov(y, f)[0, 1] / ((1 + n / N) * var_f)
            lam = float(np.clip(lam, 0, 1))
            est = y.mean() + lam * (fu.mean() - f.mean())
            var = (y - lam * f).var(ddof=1) / n + lam**2 * fu.var(ddof=1) / N
            res["ppi_pp"][0].append(est)
            res["ppi_pp"][1].append(np.sqrt(var))
        row = {"fp_share": float(share), "plug_in_true_bias": float(p_fp - p_fn)}
        for k, (e, s) in res.items():
            row[k] = summarize(e, s, prev)
        out.append(row)
    return out


# ---------------------------------------------------------------- Experiment B
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


K, T_TRUE, T_MODEL = 3.0, 1.05, 0.80   # model threshold is shifted: over-predicts Y=1


def draw_latent(n):
    z = rng.standard_normal(n)
    y = (rng.random(n) < sigmoid(K * (z - T_TRUE))).astype(float)
    g = sigmoid(K * (z - T_MODEL))          # model score
    f = (g > 0.5).astype(float)             # model label
    unc = 1 - np.abs(2 * g - 1)             # 1 = maximally unsure
    return y, f, unc


def experiment_b(pi_floor=0.002):
    y, f, unc = draw_latent(4_000_000)      # population constants
    theta, acc, pop_f_mean = y.mean(), (y == f).mean(), f.mean()
    raw_mean = unc.mean()
    budget = N_HUMAN / N_POOL

    def pi_routed(u):                        # known design: depends on X only
        return np.clip(budget * u / raw_mean, pi_floor, 1.0)

    scale = budget / pi_routed(unc).mean()   # renormalise after clipping

    res = {k: ([], []) for k in
           ("routed_human_only", "routed_naive_ppi", "routed_ipw", "uniform_ipw")}
    n_used = []
    for _ in range(REPS):
        y, f, unc = draw_latent(N_POOL)
        pi = np.clip(pi_routed(unc) * scale, pi_floor, 1.0)
        r = rng.random(N_POOL) < pi
        n_used.append(r.sum())
        yr, fr = y[r], f[r]
        res["routed_human_only"][0].append(yr.mean())
        res["routed_human_only"][1].append(yr.std(ddof=1) / np.sqrt(r.sum()))
        rect = yr - fr
        res["routed_naive_ppi"][0].append(f.mean() + rect.mean())
        res["routed_naive_ppi"][1].append(
            np.sqrt(f.var(ddof=1) / N_POOL + rect.var(ddof=1) / r.sum()))
        pseudo = f + (r / pi) * (y - f)      # DSL-style pseudo-outcome
        res["routed_ipw"][0].append(pseudo.mean())
        res["routed_ipw"][1].append(pseudo.std(ddof=1) / np.sqrt(N_POOL))
        ru = rng.random(N_POOL) < budget     # same expected budget, uniform
        pseudo_u = f + (ru / budget) * (y - f)
        res["uniform_ipw"][0].append(pseudo_u.mean())
        res["uniform_ipw"][1].append(pseudo_u.std(ddof=1) / np.sqrt(N_POOL))
    out = {"theta": float(theta), "model_accuracy": float(acc),
           "plug_in_bias": float(pop_f_mean - theta),
           "mean_humans_routed": float(np.mean(n_used))}
    for k, (e, s) in res.items():
        out[k] = summarize(e, s, theta)
    return out


if __name__ == "__main__":
    results = {"config": {"reps": REPS, "n_pool": N_POOL, "n_human": N_HUMAN},
               "experiment_a": experiment_a(), "experiment_b": experiment_b()}
    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
