"""Tight frames on a periodic 2-D frequency grid.

Every frame here is a list of real, non-negative Fourier windows W_k plus one
core window C with

    C(xi)^2 + sum_k W_k(xi)^2 = 1   at every grid frequency.

So analysis b_k = W_k x and synthesis x = C^2 x + sum_k W_k b_k is exact, and the
Sihti residues r_k = W_k(W_k x) add back to x by plain addition.

The radial part (octave bands + core) is shared by every arm. Arms differ only in
how each octave is split by direction:

    iso            1 window per octave (no direction): the Sihti/Sigh radial sieve
    fixed8         8 raised-cosine directions at every octave
    sp8            8 steerable-pyramid directions, |cos|^(K-1) (Simoncelli-Freeman)
    parabolic      raised-cosine, K = 3,4,6,8,12,15 coarse -> fine (~ 2^(j/2), curvelet law)
    antiparabolic  raised-cosine, K = 15,12,8,6,4,3 coarse -> fine (control)
    ray            the paper's window: exp(-rho |xi| |omega - xi_hat|^2 / 2),
                   normalised over the directions of the octave. One global rho, so
                   the angular width shrinks continuously as |xi|^(-1/2) inside every
                   band, not in steps.

All oriented arms have 48 bands, so they are compared at equal redundancy.
"""
import numpy as np

J = 6                                   # oriented octaves; core holds wavelengths > 64 px
COUNTS = {
    "iso": [1] * J,
    "fixed8": [8] * J,
    "sp8": [8] * J,
    "parabolic": [3, 4, 6, 8, 12, 15],
    "antiparabolic": [15, 12, 8, 6, 4, 3],
    "ray": [3, 4, 6, 8, 12, 15],
}
ARMS = list(COUNTS)

# ray arm: one imaginary translation length for the whole frame.  Chosen so that at
# the finest band centre (|xi| = 0.35 cycles/px) the Gaussian angular s.d.
# (rho |xi|)^(-1/2) matches the FWHM of a 15-direction raised cosine
# (FWHM pi/15 -> sigma = pi / (2.355 * 15)).  Frozen before any benchmark.
RAY_RHO = 1.0 / (0.35 * (np.pi / (2.355 * 15)) ** 2)


def grid(H, W):
    fy = np.fft.fftfreq(H)[:, None] * np.ones((1, W))
    fx = np.fft.fftfreq(W)[None, :] * np.ones((H, 1))
    r = np.hypot(fx, fy)
    th = np.mod(np.arctan2(fy, fx), np.pi)          # orientation in [0, pi)
    return fx, fy, r, th


def _neg(A):
    """A evaluated at -xi on the periodic grid."""
    return np.roll(np.flip(A, axis=(0, 1)), 1, axis=(0, 1))


def _lowpass(r, cutoff):
    """Smooth radial lowpass (amplitude): 1 below cutoff/2, 0 above cutoff,
    cos(pi/2 * log2(2r/cutoff)) in between."""
    t = np.clip(np.log2(np.maximum(2.0 * r / cutoff, 1e-12)), 0.0, 1.0)
    return np.cos(0.5 * np.pi * t)


def radial(H, W):
    """Squared radial octave windows B_j^2 (j = 0 finest) and the squared core C^2.
    Band j covers roughly 0.25*2^-j .. 0.5*2^-j cycles/px; band 0 also takes the
    corners up to 0.707."""
    _, _, r, _ = grid(H, W)
    lam2 = [np.ones_like(r)] + [_lowpass(r, 0.5 * 2.0 ** -j) ** 2 for j in range(J)]
    bands2 = [lam2[j] - lam2[j + 1] for j in range(J)]
    return [np.maximum(b, 0.0) for b in bands2], lam2[J]


def band_centre(j):
    """Geometric centre frequency (cycles/px) of octave j."""
    return 0.25 * 2.0 ** -j * np.sqrt(2.0) if j > 0 else 0.35


def _wrap(d):
    return (d + np.pi / 2) % np.pi - np.pi / 2


def angular(th, r, K, kind):
    """Squared angular windows P_k (sum_k P_k = 1) for one octave."""
    if K == 1:
        return [np.ones_like(th)], [0.0]
    cen = [k * np.pi / K for k in range(K)]
    if kind == "raised":
        P = []
        for c in cen:
            d = _wrap(th - c)
            P.append(np.where(np.abs(d) < np.pi / K, np.cos(0.5 * K * d) ** 2, 0.0))
    elif kind == "sp":
        A = [np.abs(np.cos(_wrap(th - c))) ** (2 * (K - 1)) for c in cen]
        P = A
    elif kind == "ray":
        # |omega - xi_hat|^2 = 2 - 2|cos(d)| for the nearer of +-omega
        P = [np.exp(-0.5 * RAY_RHO * r * (2.0 - 2.0 * np.abs(np.cos(_wrap(th - c))))) for c in cen]
    else:
        raise ValueError(kind)
    S = np.sum(P, axis=0)
    S = np.where(S > 0, S, 1.0)
    return [p / S for p in P], cen


KIND = {"iso": "raised", "fixed8": "raised", "sp8": "sp", "parabolic": "raised",
        "antiparabolic": "raised", "ray": "ray"}


class Frame:
    """Tight frame for one arm on an H x W periodic grid.

    windows : list of real arrays W_k (amplitude)
    meta    : list of dicts {j, k, K, theta}
    core2   : C^2
    """

    def __init__(self, arm, H, W):
        if arm not in COUNTS:
            raise ValueError(arm)
        self.arm, self.H, self.W = arm, H, W
        fx, fy, r, th = grid(H, W)
        self.fx, self.fy = fx, fy
        bands2, self.core2 = radial(H, W)
        counts = COUNTS[arm][::-1]                      # stored coarse->fine; j=0 is finest
        self.windows, self.meta = [], []
        for j in range(J):
            K = counts[j]
            P, cen = angular(th, r, K, KIND[arm])
            for k, (p, c) in enumerate(zip(P, cen)):
                p = 0.5 * (p + _neg(p))                 # exact Hermitian symmetry on the grid
                self.windows.append(np.sqrt(np.maximum(bands2[j] * p, 0.0)))
                self.meta.append({"j": j, "k": k, "K": K, "theta": c})
        self.oriented = arm != "iso"

    # ------------------------------------------------------------------
    def check(self):
        """max |C^2 + sum W_k^2 - 1| over the grid."""
        S = self.core2 + np.sum([w * w for w in self.windows], axis=0)
        return float(np.abs(S - 1.0).max())

    def mirror_partner(self):
        """Band index whose window is this band's window reflected fx -> -fx
        (equivalently fy -> -fy): same octave, orientation theta -> pi - theta."""
        out = []
        for m in self.meta:
            K, k = m["K"], m["k"]
            k2 = (K - k) % K
            out.append(next(i for i, n in enumerate(self.meta) if n["j"] == m["j"] and n["k"] == k2))
        return out

    def half_plane(self, idx):
        """2*U for the analytic signal of band idx (U = half plane facing its direction)."""
        th = self.meta[idx]["theta"]
        d = self.fx * np.cos(th) + self.fy * np.sin(th)
        return np.where(d > 0, 2.0, np.where(d == 0, 1.0, 0.0))
