# Pre-registration

Written and pushed **before** the confirmatory run on the BSDS500 *test* split. The
decision rules below are also frozen in `analyze.py` (`PRIMARY_BUDGET`, `DECIDED`,
`COMPARISONS`, `MATCHED`, `SECONDARY`) at this commit. Nothing on the test split has
been looked at; the 30-image discovery run used *val* only.

## Question

Sihti's G4 asked *where the objects go* when an image is sieved: into the core or the
residues. Here every octave residue is further split by **direction**, under six
direction laws at equal redundancy, and we ask: when only a budget of coefficients may
be kept, which direction law keeps the objects?

The trigger was OpenAI's local-smoothing preprint (Oct 2026). Its §1.4–2 filters waves
by direction by sliding them along an imaginary light ray, which multiplies mode ξ by
exp(−ρ|ξ||ω−ξ̂|²/2). That window's angular width shrinks as (ρ|ξ|)^(−1/2): the same
parabolic law as curvelets (Candès–Donoho 2004). The paper does **not** claim anything
about images; everything here is our test, not its consequence.

## Arms (all tight frames, all lossless: core + Σ residues = image to 1e-12)

| arm | directions per octave, coarse → fine | window |
|---|---|---|
| `iso` | 1 (Sihti/Sigh radial sieve) | — |
| `fixed8` | 8 at every octave | raised cosine |
| `sp8` | 8 at every octave | steerable pyramid, \|cos\|^(K−1) (Simoncelli–Freeman 1995) |
| `parabolic` | 3, 4, 6, 8, 12, 15 (≈ 2^(j/2), curvelet law) | raised cosine |
| `antiparabolic` | 15, 12, 8, 6, 4, 3 | raised cosine |
| `ray` | 3, 4, 6, 8, 12, 15 | the paper's window, one global ρ, normalised per octave |

All oriented arms have 48 bands. `iso` has 6, so it is **6/48 as redundant**; that
confound is the reason for the fidelity-matched tests below.

## Protocol (shared with Sihti G4)

BSDS500, long side 120 px (area resize), human segmentations nearest-resized. Images
are mirror-extended so the FFT sees no frame edge; a kept coefficient switches on its
four mirror copies and is counted once. Budget B = fraction of the 3·H·W reals; the
core (wavelengths > 64 px) is always kept and not counted. Ranking: largest |b|
("magnitude") or |b|·ρ/ρ̄ ("coherence", local cross-scale phase coherence, octaves 1–5
only; Gate 0 found octave 0 coherence at noise level, see `tests/test_gate0.py`).

Metric: Sihti's `alignment(X) = f(X, humans rotated 180°) − f(X, humans)` of the
rebuild X = core + kept detail, `within_fraction` copied unchanged from Sihti.

## Frozen choices

- Primary budget **10 %**, primary metric **G4 alignment of the rebuild**. Chosen on
  discovery: the H1 effect grows with budget (val, 30 images: +0.004 at 1 % to +0.034
  at 25 %), 10 % is the largest budget below "a quarter of the image".
- Decision rule (every decided test): mean of a > mean of b, a wins on ≥ 60 % of images,
  two-sided sign test p < 0.01. All 200 test images, none excluded.
- Fidelity-matched tests: a's alignment interpolated over its five budgets to the PSNR
  that b reaches at 10 %, same rule.

## Decided hypotheses, with the discovery numbers and our predictions

| id | claim (a beats b) | discovery (val, n = 30) | prediction |
|---|---|---|---|
| H1 | parabolic > fixed8 | 25/5, +0.020 | pass |
| H1k | parabolic > sp8 — **the kill condition**: if this fails, parabolic Sihti is a steerable pyramid with extra steps | 25/5, +0.022 | pass |
| H1m | parabolic > fixed8 **at matched PSNR** (more object structure per unit fidelity, i.e. beyond curvelet sparsity) | 19/11, +0.010 | uncertain |
| H1mk | parabolic > sp8 at matched PSNR | 21/9, +0.007 | uncertain |
| H2 | ray (paper's window) > parabolic steps | 12/18, −0.001 | **fail**: the paper's window is a curvelet window |
| H3 | coherence ranking > magnitude ranking (alignment) | 10/20, −0.016 | **fail** |
| H3b | coherence ranking > magnitude ranking on **boundary concentration** (kept detail energy within 1 px of human boundaries, minus 180°-rotated) | 20/10, +0.061 | pass |
| H0 | parabolic > iso | 5/25, −0.033 | **fail** (redundancy) |

Reported, not decided: parabolic vs antiparabolic, fixed8 vs iso, sp8 vs fixed8,
coherence on fixed8, parabolic vs iso at matched PSNR (discovery 14/16: the iso
advantage vanishes at equal fidelity).

## What would change the README

- H1k fails → the headline is "no better than a steerable pyramid" and the line closes.
- H1 and H1k pass but H1m/H1mk fail → the gain is the known curvelet sparsity result
  (Candès–Donoho), measured on objects; nothing new about where objects go.
- H1m and H1mk pass → parabolic direction puts more object structure into a rebuild of
  equal quality. This is the only outcome that would be a contribution.
- H2 passes → the paper's continuous window carries something curvelet steps do not.
- H3 fails and H3b passes → cross-scale phase coherence finds outlines, and outlines
  are not what G4 calls objects (G4 rewards filled regions).
