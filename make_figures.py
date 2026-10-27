"""Figures for the README (run after bench.py --phase test)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from posihti import Frame, Decomposition, frame_for
from bench import load, list_split

BSDS = os.environ.get("BSDS", "/home/claude/data/BSDS500/BSDS500/data")
ARM_LABEL = {"iso": "isotropic (Sihti radial)", "fixed8": "fixed 8", "sp8": "steerable pyramid 8",
             "parabolic": "parabolic 3-15", "antiparabolic": "anti-parabolic 15-3", "ray": "ray (paper's window)"}
COL = {"iso": "#888888", "fixed8": "#4c78a8", "sp8": "#72b7b2", "parabolic": "#e0a030",
       "antiparabolic": "#b279a2", "ray": "#e45756", "parabolic+coh": "#2f7d32", "fixed8+coh": "#54a24b"}


def partition_image(arm, n=256):
    fr = Frame(arm, n, n)
    W = np.array(fr.windows)
    best = W.argmax(0)
    has = W.max(0) > 1e-3
    img = np.zeros((n, n, 3))
    for k, m in enumerate(fr.meta):
        sel = (best == k) & has
        hue = (m["k"] / m["K"] + (m["j"] % 2) * 0.5 / m["K"]) % 1 if fr.oriented else 0.0
        img[sel] = hsv_to_rgb([hue, 0.65 if fr.oriented else 0.0, 0.45 + 0.09 * m["j"]])
    img[~has] = 0.95
    return np.fft.fftshift(img, axes=(0, 1))


def fig_partitions():
    arms = ["iso", "fixed8", "sp8", "parabolic", "ray", "antiparabolic"]
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.4))
    for a, arm in zip(ax, arms):
        a.imshow(partition_image(arm), extent=[-.5, .5, -.5, .5])
        a.set_title(ARM_LABEL[arm], fontsize=11)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Frequency plane, each cell coloured by its dominant band (hue = direction, lightness = octave). "
                 "All oriented arms have 48 bands.", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/partitions.png", dpi=110)
    plt.close(fig)


def fig_curves(T, path="figures/budget_curves.png"):
    means = T["means"]
    budgets = sorted(means["parabolic"], key=float)
    xs = [100 * float(b) for b in budgets]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for arm in ["iso", "fixed8", "sp8", "parabolic", "ray", "antiparabolic", "parabolic+coh"]:
        if arm not in means:
            continue
        ls = "--" if arm.endswith("+coh") else "-"
        for a, m in zip(ax, ["align", "bc", "psnr"]):
            a.plot(xs, [means[arm][b][m] for b in budgets], ls, marker="o", ms=4, color=COL[arm],
                   label=ARM_LABEL.get(arm, arm.replace("+coh", " + coherence ranking")))
    ax[0].axhline(T["ref"]["align_core"], color="k", lw=.8, ls=":")
    ax[0].text(xs[0], T["ref"]["align_core"], " core only", va="bottom", fontsize=9)
    ax[0].axhline(T["ref"]["align_image"], color="k", lw=.8, ls=":")
    ax[0].text(xs[0], T["ref"]["align_image"], " whole image", va="bottom", fontsize=9)
    for a, t in zip(ax, ["G4 alignment of the rebuild (primary)", "kept detail on human boundaries",
                         "PSNR of the rebuild (dB)"]):
        a.set_xscale("log"); a.set_xlabel("budget, % of image reals"); a.set_title(t, fontsize=11)
        a.set_xticks(xs); a.set_xticklabels([f"{x:g}" for x in xs])
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"BSDS500 test, n = {T['images']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_examples(names, budget, path="figures/examples.png"):
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in list_split(BSDS, "test")}
    cols = ["image", "fixed8", "parabolic", "parabolic+coh", "coherence"]
    fig, ax = plt.subplots(len(names), len(cols), figsize=(3.0 * len(cols), 2.3 * len(names)))
    for r, name in enumerate(names):
        _, x, gts = load(BSDS, "test", paths[name])
        n = int(budget * x.size)
        ax[r, 0].imshow(x)
        for c, arm in enumerate(["fixed8", "parabolic"]):
            d = Decomposition(frame_for(arm, x), x)
            ax[r, c + 1].imshow(np.clip(d.synthesize(d.mterm_mask(n)), 0, 1))
        d = Decomposition(frame_for("parabolic", x), x)
        rho = d.coherence()
        ax[r, 3].imshow(np.clip(d.synthesize(d.mterm_mask(n, "coherence", rho)), 0, 1))
        H, W = x.shape[:2]
        num = np.zeros((H, W)); den = np.zeros((H, W))
        for k, m in enumerate(d.frame.meta):
            if m["j"] == 0:
                continue
            for c in range(3):
                e = d.bands[c][k][:H, :W] ** 2
                num += e * rho[c][k][:H, :W]; den += e
        ax[r, 4].imshow(num / np.maximum(den, 1e-12), cmap="magma", vmin=0, vmax=1)
    titles = ["image", f"fixed 8, {100 * budget:g}%", f"parabolic, {100 * budget:g}%",
              f"parabolic + coherence rank", "cross-scale coherence (oct 1-5)"]
    for c, t in enumerate(titles):
        ax[0, c].set_title(t, fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    fig_partitions()
    if os.path.exists("results/TEST.json"):
        T = json.load(open("results/TEST.json"))
        fig_curves(T)
        fig_examples(["100075", "101087", "103078", "104022"], T["decisions"]["budget"])
