"""Analysis, lossless residues, m-term reconstruction and cross-scale coherence.

Images are H x W x C in [0, 1].  Each channel is mirror-extended to 2H x 2W
(even reflection) so the periodic FFT sees no frame-edge jump; every coefficient
lives on that extended grid, but only those in the original H x W quadrant are
ever counted, ranked or kept.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .frames import Frame, J, band_centre


def mirror(x2d):
    top = np.concatenate([x2d, x2d[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


class Decomposition:
    """All bands of one image under one arm.

    bands[c][k]  real band k of channel c on the extended grid
    core[c]      core field (C^2 x) of channel c on the extended grid
    """

    def __init__(self, frame, img):
        img = np.asarray(img, dtype=np.float64)
        if img.ndim == 2:
            img = img[..., None]
        self.frame, self.H, self.W, self.C = frame, img.shape[0], img.shape[1], img.shape[2]
        assert frame.H == 2 * self.H and frame.W == 2 * self.W
        self.F = [np.fft.fft2(mirror(img[..., c])) for c in range(self.C)]
        self.bands = [[np.real(np.fft.ifft2(w * F)) for w in frame.windows] for F in self.F]
        self.core = [np.real(np.fft.ifft2(frame.core2 * F)) for F in self.F]
        self._analytic = None

    # -- Sihti view: lossless residues ---------------------------------
    def residues(self):
        """r_k = W_k(W_k x), cropped; sum_k r_k + core == image exactly."""
        out = []
        for k, w in enumerate(self.frame.windows):
            r = np.stack([np.real(np.fft.ifft2(w * w * F))[:self.H, :self.W] for F in self.F], -1)
            out.append(r)
        core = np.stack([c[:self.H, :self.W] for c in self.core], -1)
        return out, core

    def synthesize(self, mask=None):
        """core + sum_k W_k (mask_k * b_k), cropped.  mask=None keeps everything."""
        out = []
        for c in range(self.C):
            acc = np.fft.fft2(self.core[c])
            for k, w in enumerate(self.frame.windows):
                b = self.bands[c][k]
                if mask is not None:
                    b = np.where(mask[c][k], b, 0.0)
                acc = acc + w * np.fft.fft2(b)
            out.append(np.real(np.fft.ifft2(acc))[:self.H, :self.W])
        return np.stack(out, -1)

    def core_image(self):
        return np.stack([c[:self.H, :self.W] for c in self.core], -1)

    # -- analytic bands and the cross-term --------------------------------
    def analytic(self):
        if self._analytic is None:
            fr = self.frame
            U = [fr.half_plane(k) for k in range(len(fr.windows))]
            self._analytic = [[np.fft.ifft2(U[k] * w * F) for k, w in enumerate(fr.windows)]
                              for F in self.F]
        return self._analytic

    def coherence(self):
        """Local cross-scale phase coherence rho_k(x) in [0, 1] for every oriented band.

        Pair band (j, theta) with the nearest-angle band of the adjacent octave
        (the coarser one; the coarsest octave pairs with the finer one).  With p the
        coarser analytic band and a the finer,

            p2 = p^2 / |p|            (phase doubled: same frequency as a)
            q  = p2 conj a            (constant phase across an edge: phase congruency)
            rho = |G * (q^2/|q|)| / (G * |q|)    (squared: polarity-invariant)

        G a Gaussian of s.d. one fine-band wavelength.  An edge that is phase-aligned
        across the two octaves gives rho -> 1; texture with scrambled phase -> 0.
        Pooled over the whole image the cross term would vanish for disjoint bands;
        it is only ever used locally.

        Squaring p doubles its frequency, so the finest pair would fold over Nyquist.
        Everything is therefore computed on a 2x upsampled grid (zero-padded spectra)
        and sampled back.
        """
        fr = self.frame
        if not fr.oriented:
            raise ValueError("coherence needs oriented bands")
        N1, N2 = fr.H, fr.W

        def up(S):
            out = np.zeros((2 * N1, 2 * N2), dtype=complex)
            h1, h2 = N1 // 2, N2 // 2
            out[:h1, :h2] = S[:h1, :h2]
            out[:h1, -(N2 - h2):] = S[:h1, h2:]
            out[-(N1 - h1):, :h2] = S[h1:, :h2]
            out[-(N1 - h1):, -(N2 - h2):] = S[h1:, h2:]
            return np.fft.ifft2(out) * 4.0

        cache = {}

        def analytic_up(c, k):
            if (c, k) not in cache:
                cache[(c, k)] = up(fr.half_plane(k) * fr.windows[k] * self.F[c])
            return cache[(c, k)]
        by_j = {}
        for idx, m in enumerate(fr.meta):
            by_j.setdefault(m["j"], []).append(idx)

        def nearest(idx, j2):
            th = fr.meta[idx]["theta"]
            cands = by_j[j2]
            d = [abs(((fr.meta[c]["theta"] - th) + np.pi / 2) % np.pi - np.pi / 2) for c in cands]
            return cands[int(np.argmin(d))]

        rho = [[None] * len(fr.windows) for _ in range(self.C)]
        for idx, m in enumerate(fr.meta):
            j = m["j"]
            if j + 1 < J:                       # finer band; parent is octave j+1
                fine, coarse, jf = idx, nearest(idx, j + 1), j
            else:                               # coarsest octave: pair with child
                fine, coarse, jf = nearest(idx, j - 1), idx, j - 1
            sig = 2.0 / band_centre(jf)                  # one wavelength, on the 2x grid
            for c in range(self.C):
                a, p = analytic_up(c, fine), analytic_up(c, coarse)
                p2 = p * p / np.maximum(np.abs(p), 1e-12)
                q = p2 * np.conj(a)
                # q flips sign with edge polarity (a -> -a, p2 unchanged), so
                # opposite-polarity edges in one window would cancel.  Average
                # q^2/|q| instead: polarity-invariant, same magnitude units.
                num = q * q / np.maximum(np.abs(q), 1e-12)
                nr = gaussian_filter(num.real, sig, mode="wrap")
                ni = gaussian_filter(num.imag, sig, mode="wrap")
                den = gaussian_filter(np.abs(q), sig, mode="wrap")
                rho[c][idx] = (np.hypot(nr, ni) / np.maximum(den, 1e-12))[::2, ::2]
            if len(cache) > 6 * self.C:                   # bound memory
                cache.clear()
        return rho

    # -- m-term ------------------------------------------------------------
    def mterm_mask(self, n_keep, score="magnitude", rho=None):
        """Keep the n_keep coefficients (reals, inside the original quadrant) with the
        largest score.

        magnitude  : |b|
        coherence  : |b| * rho / rho_bar_j, rho_bar_j the |b|-weighted mean of rho over
                     octave j of that channel.  Dividing by the octave mean keeps the
                     split of the budget between octaves the same as magnitude ranking,
                     so coherence only decides WHICH coefficients inside an octave are
                     kept (edge-like vs texture-like), not how much each octave gets.
                     Octave 0 (finest) stays magnitude-ranked: Gate 0 measured its
                     coherence at noise level even on a clean antialiased edge
                     (0.22 vs 0.14 for noise; octaves 1-5: 0.66-0.83 vs 0.10-0.17)."""
        H, W = self.H, self.W
        if score not in ("magnitude", "coherence"):
            raise ValueError(score)
        scores = []
        for c in range(self.C):
            mags = [np.abs(self.bands[c][k][:H, :W]) for k in range(len(self.frame.windows))]
            if score == "coherence":
                num, den = {}, {}
                for k, m in enumerate(self.frame.meta):
                    r = rho[c][k][:H, :W]
                    num[m["j"]] = num.get(m["j"], 0.0) + (r * mags[k]).sum()
                    den[m["j"]] = den.get(m["j"], 0.0) + mags[k].sum()
                rbar = {j: num[j] / max(den[j], 1e-300) for j in num}
            for k, m in enumerate(self.frame.meta):
                s = mags[k]
                if score == "coherence" and m["j"] > 0:
                    s = s * rho[c][k][:H, :W] / max(rbar[m["j"]], 1e-12)
                scores.append(s)
        flat = np.concatenate([s.ravel() for s in scores])
        n_keep = int(min(max(n_keep, 0), flat.size))
        if n_keep == 0:
            thr = np.inf
        else:
            thr = np.partition(flat, flat.size - n_keep)[flat.size - n_keep]
        nb = len(self.frame.windows)
        part = self.frame.mirror_partner()
        mask = []
        for c in range(self.C):
            q = [scores[c * nb + k] >= thr for k in range(nb)]
            row = []
            for k in range(nb):
                # The extended image is mirror-symmetric, so band k's coefficient at a
                # mirrored position IS band partner(k)'s coefficient at the original
                # position (a flip maps orientation theta -> pi - theta).  One chosen
                # coefficient therefore switches on all four of its copies; it is
                # counted once.
                p = part[k]
                top = np.concatenate([q[k], q[p][:, ::-1]], axis=1)
                bot = np.concatenate([q[p][::-1, :], q[k][::-1, ::-1]], axis=1)
                row.append(np.concatenate([top, bot], axis=0))
            mask.append(row)
        return mask


def frame_for(arm, img):
    H, W = img.shape[:2]
    return Frame(arm, 2 * H, 2 * W)
