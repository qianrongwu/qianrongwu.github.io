"""Why does PPI++ show a small negative mean error in Experiment A?
Compare: oracle lambda, estimated lambda (same labels, unclipped / clipped to [0,1]), 2-fold cross-fitted lambda."""
import json
import numpy as np
import simulate
from simulate import draw_confusion, N_HUMAN as n, N_POOL as N, Z

SEED = 20260921
simulate.rng = np.random.default_rng(SEED)   # draw_confusion uses simulate.rng
out = {"seed": SEED, "reps": 20000, "n_human": n, "n_pool": N, "settings": []}
rng_reps = 20000
prev, err = 0.20, 0.12
for share in (0.0, 0.5, 1.0):
    p_fp, p_fn = err*share, err*(1-share); fpr, fnr = p_fp/(1-prev), p_fn/prev
    # oracle lambda from population moments
    pf = prev*(1-fnr) + (1-prev)*fpr; cov = prev*(1-fnr) - prev*pf; lam_or = cov/((1+n/N)*pf*(1-pf))
    est = {k: [] for k in ("oracle", "same_unclipped", "same_clipped", "crossfit")}
    cover = {k: 0 for k in est}
    for _ in range(rng_reps):
        y, f = draw_confusion(n, prev, fpr, fnr); _, fu = draw_confusion(N, prev, fpr, fnr)
        def ppipp(lam, yy=y, ff=f):
            e = yy.mean() + lam*(fu.mean() - ff.mean())
            v = (yy - lam*ff).var(ddof=1)/len(yy) + lam**2*fu.var(ddof=1)/N
            return e, np.sqrt(v)
        def lam_hat(yy, ff):
            vf = np.concatenate([ff, fu]).var(ddof=1)
            return np.cov(yy, ff)[0, 1]/((1+len(yy)/N)*vf)
        l = lam_hat(y, f)
        h = n//2
        l1, l2 = lam_hat(y[:h], f[:h]), lam_hat(y[h:], f[h:])
        e1, s1 = ppipp(l2, y[:h], f[:h]); e2, s2 = ppipp(l1, y[h:], f[h:])
        res = {"oracle": ppipp(lam_or), "same_unclipped": ppipp(l),
               "same_clipped": ppipp(float(np.clip(l, 0, 1))),
               "crossfit": ((e1+e2)/2, np.sqrt(s1**2+s2**2)/2)}
        for k, (e, s) in res.items():
            est[k].append(e); cover[k] += abs(e-prev) <= Z*s
    print(f"FP share {share}: oracle lambda={lam_or:.3f}")
    row = {"fp_share": share, "oracle_lambda": lam_or, "variants": {}}
    for k, v in est.items():
        v = np.array(v)
        row["variants"][k] = {"mean_error_pp": float(100*(v.mean()-prev)), "mc_se_pp": float(100*v.std()/np.sqrt(len(v))), "coverage_pct": float(100*cover[k]/rng_reps)}
        print(f"  {k:15s} mean error {100*(v.mean()-prev):+.3f} pp (MC se {100*v.std()/np.sqrt(len(v)):.3f})  coverage {100*cover[k]/rng_reps:.1f}%")
    out["settings"].append(row)
json.dump(out, open("ppipp_diagnostic_results.json", "w"), indent=2)
