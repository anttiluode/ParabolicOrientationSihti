"""Summaries and the pre-registered decisions (rules frozen in PREREG.md)."""
import numpy as np
from scipy.stats import binomtest, wilcoxon

METRICS = ["align", "align_det", "bc", "psnr"]
PRIMARY_METRIC = "align"
PRIMARY_BUDGET = "0.1"         # frozen in PREREG.md after the discovery run (commit before test)
WIN_FRACTION = 0.60
ALPHA = 0.01

# (name, a, b): "a beats b"
COMPARISONS = [
    ("H1_parabolic_vs_fixed8", "parabolic", "fixed8"),
    ("H1k_parabolic_vs_sp8", "parabolic", "sp8"),
    ("H2_ray_vs_parabolic", "ray", "parabolic"),
    ("H3_coherence_vs_magnitude", "parabolic+coh", "parabolic"),
    ("H0_parabolic_vs_iso", "parabolic", "iso"),
    ("ctl_parabolic_vs_antiparabolic", "parabolic", "antiparabolic"),
    ("ctl_fixed8_vs_iso", "fixed8", "iso"),
    ("ctl_sp8_vs_fixed8", "sp8", "fixed8"),
    ("ctl_fixed8_coherence_vs_magnitude", "fixed8+coh", "fixed8"),
]


# Fidelity-matched: arm a's alignment interpolated (over its 5 budgets) to the PSNR that
# arm b reaches at the primary budget.  Asks whether a puts more object structure into
# the rebuild than b *at the same reconstruction quality*, i.e. beyond sparsity.
MATCHED = [
    ("H1m_parabolic_vs_fixed8_matched", "parabolic", "fixed8"),
    ("H1mk_parabolic_vs_sp8_matched", "parabolic", "sp8"),
    ("H0m_parabolic_vs_iso_matched", "parabolic", "iso"),
]
# Secondary metric decisions: (name, a, b, metric)
SECONDARY = [
    ("H3b_coherence_moves_budget_to_boundaries", "parabolic+coh", "parabolic", "bc"),
]
DECIDED = ["H1_parabolic_vs_fixed8", "H1k_parabolic_vs_sp8", "H1m_parabolic_vs_fixed8_matched",
           "H1mk_parabolic_vs_sp8_matched", "H2_ray_vs_parabolic", "H3_coherence_vs_magnitude",
           "H3b_coherence_moves_budget_to_boundaries", "H0_parabolic_vs_iso"]


def matched_pair(rows, a, b, budget):
    xa, xb, clamped = [], [], 0
    for r in rows:
        buds = list(r[a].keys())
        pa = np.array([r[a][x]["psnr"] for x in buds])
        aa = np.array([r[a][x]["align"] for x in buds])
        t = r[b][budget]["psnr"]
        o = np.argsort(pa)
        clamped += int(t < pa.min() or t > pa.max())
        xa.append(float(np.interp(t, pa[o], aa[o])))
        xb.append(r[b][budget]["align"])
    out = paired(xa, xb)
    out["clamped"] = clamped
    return out


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    wins, losses = int((a > b).sum()), int((a < b).sum())
    p_sign = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    try:
        p_w = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p_w = 1.0
    diff = a - b
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "wins": wins, "losses": losses, "ties": int(len(a) - wins - losses),
            "win_fraction": float(wins / len(a)), "p_sign": float(p_sign), "p_wilcoxon": p_w}


def arms_of(rows):
    return [k for k in rows[0] if k not in ("image", "shape", "ref")]


def summarize(rows, decide, primary_budget=None):
    primary_budget = primary_budget or PRIMARY_BUDGET
    arms = arms_of(rows)
    budgets = list(rows[0][arms[0]].keys())
    out = {"images": len(rows),
           "ref": {k: float(np.mean([r["ref"][k] for r in rows])) for k in rows[0]["ref"]},
           "means": {arm: {b: {m: float(np.mean([r[arm][b][m] for r in rows])) for m in METRICS}
                           for b in budgets} for arm in arms},
           "comparisons": {}}
    for name, a, b in COMPARISONS:
        if a not in arms or b not in arms:
            continue
        out["comparisons"][name] = {
            bud: {m: paired([r[a][bud][m] for r in rows], [r[b][bud][m] for r in rows]) for m in METRICS}
            for bud in budgets}
    out["comparisons_secondary"] = {
        name: paired([r[a][primary_budget][m] for r in rows], [r[b][primary_budget][m] for r in rows])
        for name, a, b, m in SECONDARY if a in arms and b in arms}
    out["matched"] = {name: matched_pair(rows, a, b, primary_budget)
                      for name, a, b in MATCHED if a in arms and b in arms}
    if decide:
        if primary_budget is None:
            raise RuntimeError("PRIMARY_BUDGET must be frozen before the test run")
        rule = lambda c: bool(c["mean_a"] > c["mean_b"] and c["win_fraction"] >= WIN_FRACTION
                              and c["p_sign"] < ALPHA)
        allc = {}
        for name, _, _ in COMPARISONS:
            allc[name] = out["comparisons"][name][primary_budget][PRIMARY_METRIC]
        for name, a, b, metric in SECONDARY:
            allc[name] = out["comparisons_secondary"][name]
        allc.update(out["matched"])
        dec = {}
        for name, c in allc.items():
            dec[name] = {"pass": rule(c), "decided": name in DECIDED, **c}
        out["decisions"] = {"budget": primary_budget, "metric": PRIMARY_METRIC,
                            "rule": f"mean higher, wins >= {WIN_FRACTION:.0%} of images, sign test p < {ALPHA}",
                            **dec}
    return out
