"""The Sihti G4 measure and two companions.

within_fraction and alignment are copied unchanged from anttiluode/Sihti
(sihti/sieve.py and bench_bsds.py), so numbers here are on the same scale as
Sihti's published G4 results.
"""
import numpy as np
from scipy.ndimage import binary_dilation


def within_fraction(R, labels):
    """How much of a field's energy varies *inside* the segments of `labels`.
    0 = flat on every segment (moves whole regions), 1 = segments explain none of it."""
    C = R.shape[-1] if R.ndim == 3 else 1
    X = R.reshape(-1, C)
    lab = np.asarray(labels).ravel()
    _, lab = np.unique(lab, return_inverse=True)
    total = ((X - X.mean(axis=0)) ** 2).sum()
    if total <= 0:
        return 0.0
    counts = np.bincount(lab).astype(np.float64)
    between = 0.0
    for c in range(C):
        s = np.bincount(lab, weights=X[:, c])
        between += (s * s / np.maximum(counts, 1)).sum() - X[:, c].sum() ** 2 / X.shape[0]
    return float(max(0.0, total - between) / total)


def alignment(X, gts):
    """Sihti G4: f(X, humans rotated 180 deg) - f(X, humans).  A merely smooth field
    scores ~0; only fields shaped like the human segments score high."""
    return float(np.mean([within_fraction(X, np.rot90(g, 2)) - within_fraction(X, g) for g in gts]))


def boundary_map(g, width=1):
    b = np.zeros(g.shape, dtype=bool)
    b[:, :-1] |= g[:, :-1] != g[:, 1:]
    b[:-1, :] |= g[:-1, :] != g[1:, :]
    return binary_dilation(b, iterations=width) if width else b


def boundary_concentration(D, gts, width=1):
    """Fraction of D's energy within `width` px of human boundaries, minus the same
    for the 180-degree-rotated humans (same boundary density, wrong places)."""
    E = (D ** 2).sum(axis=-1) if D.ndim == 3 else D ** 2
    tot = E.sum()
    if tot <= 0:
        return 0.0
    vals = []
    for g in gts:
        b, br = boundary_map(g, width), boundary_map(np.rot90(g, 2), width)
        vals.append(E[b].sum() / tot - E[br].sum() / tot)
    return float(np.mean(vals))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
