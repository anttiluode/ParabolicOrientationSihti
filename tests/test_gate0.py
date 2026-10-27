"""Gate 0: the identities every later number depends on."""
import numpy as np
import pytest

from posihti import ARMS, COUNTS, Frame, Decomposition, frame_for, alignment, J

RNG = np.random.default_rng(0)
IMG = RNG.random((40, 56, 3))


@pytest.mark.parametrize("arm", ARMS)
def test_tight_frame(arm):
    assert Frame(arm, 80, 112).check() < 1e-12


@pytest.mark.parametrize("arm", ARMS)
def test_equal_redundancy(arm):
    n = len(Frame(arm, 32, 32).windows)
    assert n == (J if arm == "iso" else 48)
    assert sum(COUNTS[arm]) == n


@pytest.mark.parametrize("arm", ARMS)
def test_lossless_residues_and_synthesis(arm):
    d = Decomposition(frame_for(arm, IMG), IMG)
    res, core = d.residues()
    assert np.abs(sum(res) + core - IMG).max() < 1e-12          # Sihti identity: plain addition
    assert np.abs(d.synthesize() - IMG).max() < 1e-12           # frame synthesis


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "iso"])
def test_analytic_real_part_is_band(arm):
    d = Decomposition(frame_for(arm, IMG), IMG)
    A = d.analytic()
    err = max(np.abs(A[0][k].real - d.bands[0][k]).max() for k in range(len(d.bands[0])))
    scale = max(np.abs(b).max() for b in d.bands[0])
    assert err < 1e-2 * scale                                   # only Nyquist-line slack


def test_bands_are_real():
    fr = frame_for("parabolic", IMG)
    for w in fr.windows:
        assert np.abs(w - np.roll(np.flip(w, (0, 1)), 1, (0, 1))).max() < 1e-12


def test_coherence_bounds_and_edge_vs_noise():
    """Octaves 1-5 must separate a clean (antialiased) edge from noise.  Octave 0 does
    not (recorded in PREREG.md), so it is excluded here and from coherence ranking."""
    from scipy.special import erf
    H = W = 96
    yy, xx = np.mgrid[:H, :W]
    dist = (xx - W / 2 - 0.3 * (yy - H / 2)) / np.hypot(1, 0.3)
    edge = (0.5 + 0.5 * erf(dist / 0.7))[..., None]
    noise = RNG.standard_normal((H, W, 1)) * 0.3
    out = {}
    for name, im in (("edge", edge), ("noise", noise)):
        d = Decomposition(frame_for("parabolic", im), im)
        rho = d.coherence()
        per = {}
        for k, m in enumerate(d.frame.meta):
            r, e = rho[0][k][:H, :W], np.abs(d.bands[0][k][:H, :W])
            assert r.min() >= -1e-9 and r.max() <= 1 + 1e-9
            a = per.setdefault(m["j"], [0.0, 0.0])
            a[0] += (r * e).sum()
            a[1] += e.sum()
        out[name] = {j: v[0] / v[1] for j, v in per.items()}
    for j in range(1, J):
        assert out["edge"][j] > out["noise"][j] + 0.2, (j, out)


def test_mterm_budget_and_monotone():
    d = Decomposition(frame_for("parabolic", IMG), IMG)
    full = 3 * 40 * 56
    errs = []
    for n in (0, full // 50, full // 10, 48 * full):
        m = d.mterm_mask(n)
        kept = sum(int(x[:40, :56].sum()) for row in m for x in row)   # counted once
        assert kept >= min(n, 48 * full) and (n == 0 or kept <= n + 2)
        errs.append(np.abs(d.synthesize(m) - IMG).mean())
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-12


@pytest.mark.parametrize("arm", ARMS)
def test_mirror_partner_is_the_mirrored_band(arm):
    d = Decomposition(frame_for(arm, IMG), IMG)
    H, W = IMG.shape[:2]
    part = d.frame.mirror_partner()
    scale = max(np.abs(b).max() for b in d.bands[0])
    for k, p in enumerate(part):
        b, bp = d.bands[0][k], d.bands[0][p]
        right = b[:H, W:][:, ::-1]                  # band k in the x-mirrored quadrant
        assert np.abs(right - bp[:H, :W]).max() < 2e-2 * scale   # Nyquist column only
        both = b[H:, W:][::-1, ::-1]
        assert np.abs(both - b[:H, :W]).max() < 1e-10


def test_alignment_smooth_field_near_zero():
    g = np.zeros((40, 56), int)
    g[:, 30:] = 1
    g[25:, :] += 2
    smooth = np.add.outer(np.linspace(0, 1, 40), np.linspace(0, 1, 56))[..., None]
    shaped = g[..., None].astype(float)
    assert alignment(shaped, [g]) > 0.5
    assert abs(alignment(smooth, [g])) < 0.25
