"""Where do the objects go when the sieve has a direction law?  (see PREREG.md)

    python bench.py --bsds BSDS500/data --phase discovery   # 30 val images, report only
    python bench.py --bsds BSDS500/data --phase test        # confirmatory, decides

Protocol shared with anttiluode/Sihti G4: BSDS500, every image area-resized to a
120-px long side, human segmentations nearest-resized to the same grid, and the
Sihti alignment measure  alignment(X) = f(X, humans rotated 180) - f(X, humans).

For every arm and budget B (a fraction of the 3*H*W reals of the image) keep the B
largest band coefficients (inside the original image; core always kept, not
counted) and rebuild  X_B = core + kept detail.  Report, per image:

    align       Sihti G4 alignment of X_B            (primary)
    align_det   alignment of the kept detail X_B - core
    bc          boundary concentration of the kept detail
    psnr        X_B vs the image
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
import scipy.io as sio

from posihti import ARMS, Decomposition, frame_for, alignment, boundary_concentration, psnr

LONG = 120
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.25]
COHERENCE_ARMS = ["parabolic", "fixed8"]
DISCOVERY_N = 30


def resize_to(rgb, long_side):
    """Area-average resize (identical to sihti.pipeline.resize_to)."""
    H, W = rgb.shape[:2]
    s = float(long_side) / max(H, W)
    h, w = max(2, int(round(H * s))), max(2, int(round(W * s)))
    im = Image.fromarray(np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((w, h), Image.Resampling.BOX), dtype=np.float64) / 255.0


def list_split(root, split):
    return sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))


def load(root, split, path):
    name = os.path.splitext(os.path.basename(path))[0]
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    x = resize_to(rgb, LONG)
    h, w = x.shape[:2]
    m = sio.loadmat(os.path.join(root, "groundTruth", split, name + ".mat"))["groundTruth"]
    gts = []
    for i in range(m.shape[1]):
        s = m[0, i]["Segmentation"][0, 0].astype(np.int32)
        gts.append(np.asarray(Image.fromarray(s).resize((w, h), Image.Resampling.NEAREST)))
    return name, x, gts


def measure(d, x, gts, n_keep, score="magnitude", rho=None):
    core = d.core_image()
    X = d.synthesize(d.mterm_mask(n_keep, score, rho))
    D = X - core
    return {"align": alignment(X, gts), "align_det": alignment(D, gts),
            "bc": boundary_concentration(D, gts), "psnr": psnr(X, x)}


def run_image(args):
    root, split, path = args
    name, x, gts = load(root, split, path)
    H, W = x.shape[:2]
    reals = 3 * H * W
    row = {"image": name, "shape": [H, W],
           "ref": {"align_image": alignment(x, gts)}}
    core_done = False
    for arm in ARMS:
        d = Decomposition(frame_for(arm, x), x)
        if not core_done:
            row["ref"]["align_core"] = alignment(d.core_image(), gts)
            core_done = True
        row[arm] = {str(b): measure(d, x, gts, int(b * reals)) for b in BUDGETS}
        if arm in COHERENCE_ARMS:
            rho = d.coherence()
            row[arm + "+coh"] = {str(b): measure(d, x, gts, int(b * reals), "coherence", rho)
                                 for b in BUDGETS}
    return row


def run(root, split, paths, cache_path, workers):
    done = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            done[r["image"]] = r
    todo = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in done]
    t0 = time.time()
    with open(cache_path, "a") as fc, Pool(workers) as pool:
        for r in pool.imap_unordered(run_image, [(root, split, p) for p in todo]):
            done[r["image"]] = r
            fc.write(json.dumps(r) + "\n")
            fc.flush()
            print(f"{split} {len(done)}/{len(paths)} {r['image']}  {time.time() - t0:.0f}s", flush=True)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return [done[n] for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsds", required=True)
    ap.add_argument("--phase", choices=["discovery", "test"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if a.phase == "discovery":
        paths = list_split(a.bsds, "val")[:DISCOVERY_N]
        rows = run(a.bsds, "val", paths, "results/discovery_cache.jsonl", a.workers)
        from analyze import summarize
        json.dump(summarize(rows, decide=False), open("results/DISCOVERY.json", "w"), indent=1)
    else:
        paths = list_split(a.bsds, "test")
        rows = run(a.bsds, "test", paths, "results/test_cache.jsonl", a.workers)
        from analyze import summarize
        json.dump(summarize(rows, decide=True), open("results/TEST.json", "w"), indent=1)


if __name__ == "__main__":
    main()
