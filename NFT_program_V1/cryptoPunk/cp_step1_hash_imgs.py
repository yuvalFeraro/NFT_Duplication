"""
cp_step1_hash_imgs.py
=====================
Compute image hashes for 400 randomly selected images from
data_sets/imgs/ (these are used as extra base images and negatives
for the CryptoPunk minimal-distance detector experiment).

  200 imgs  --> added to CryptoPunk base tree
  200 imgs  --> used as negatives (queried against base)

CryptoPunk originals and manipulations already have hashes in the xlsx.

Output: NFT_program_V1/cryptoPunk/data/imgs_hashes.csv
  Columns: filename, split (base|negative), aHash, pHash, hsvHash, sHash

Run:
    python cp_step1_hash_imgs.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CP_DIR     = SCRIPT_DIR                    # NFT_program_V1/cryptoPunk/
DATA_DIR   = CP_DIR / "data"

IMGS_DIR   = Path(r"C:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets\imgs")
OUT_CSV    = DATA_DIR / "imgs_hashes.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
N_BASE        = 200     # imgs added to base tree
N_NEGATIVES   = 200     # imgs used as negatives
N_TOTAL       = N_BASE + N_NEGATIVES   # 400


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def compute_hashes(img_path: Path) -> dict[str, str]:
    """Compute all 4 hashes and return as hex strings."""
    img = Image.open(img_path).convert("RGB")

    ahash = imagehash.average_hash(img, hash_size=8)
    phash = imagehash.phash(img, hash_size=8)
    shash = imagehash.crop_resistant_hash(img)

    # hsvHash: convert to HSV, compute average hash on each channel
    img_arr = np.array(img).astype(float) / 255.0
    r, g, b  = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin + 1e-8
    h = np.zeros_like(cmax)
    mask_r = (cmax == r); h[mask_r] = 60 * ((g[mask_r]-b[mask_r]) / diff[mask_r] % 6)
    mask_g = (cmax == g); h[mask_g] = 60 * ((b[mask_g]-r[mask_g]) / diff[mask_g] + 2)
    mask_b = (cmax == b); h[mask_b] = 60 * ((r[mask_b]-g[mask_b]) / diff[mask_b] + 4)
    s = np.where(cmax == 0, 0, diff / (cmax + 1e-8))
    v = cmax
    # Divide hue into 14 bins (0-360 in 14 segments), sat and val into 3 bins each
    # → 14×3 = 42 bits stored as a 42-bit integer → hex
    n_hue_bins = 14; n_sv_bins = 3
    hue_bins = np.clip((h / 360 * n_hue_bins).astype(int), 0, n_hue_bins - 1)
    sat_bins = np.clip((s * n_sv_bins).astype(int), 0, n_sv_bins - 1)
    val_bins = np.clip((v * n_sv_bins).astype(int), 0, n_sv_bins - 1)
    bits = np.zeros(n_hue_bins * n_sv_bins, dtype=bool)
    for hb in range(n_hue_bins):
        for svb in range(n_sv_bins):
            mask = (hue_bins == hb) & (sat_bins == svb)
            bits[hb * n_sv_bins + svb] = mask.mean() > 0.05
    bit_int = int("".join("1" if b else "0" for b in bits), 2)
    hsv_hex = format(bit_int, f"0{(n_hue_bins * n_sv_bins + 3) // 4}x")

    return {
        "aHash":   str(ahash),
        "pHash":   str(phash),
        "hsvHash": hsv_hex,
        "sHash":   ",".join(str(h) for h in shash.segment_hashes),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CP STEP 1 — Hash 400 imgs/ images")
    print("=" * 60)

    # Collect all available imgs
    all_imgs = sorted(IMGS_DIR.glob("*.png"), key=lambda p: int(p.stem))
    print(f"\nAvailable imgs: {len(all_imgs)}")
    if len(all_imgs) < N_TOTAL:
        print(f"ERROR: need {N_TOTAL} images but only {len(all_imgs)} available.")
        sys.exit(1)

    # Random sample 400
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(all_imgs, N_TOTAL)
    base_imgs = selected[:N_BASE]
    neg_imgs  = selected[N_BASE:]

    print(f"Selected {N_BASE} for base + {N_NEGATIVES} for negatives (seed={RANDOM_SEED})")

    # Compute hashes
    rows = []
    total = len(selected)
    t0 = time.time()
    for i, (path, split) in enumerate(
        [(p, "base") for p in base_imgs] + [(p, "negative") for p in neg_imgs]
    ):
        try:
            h = compute_hashes(path)
            rows.append({"filename": path.name, "split": split, **h})
        except Exception as e:
            print(f"  WARNING: failed to hash {path.name}: {e}")

        if (i + 1) % 50 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            print(f"  {i+1}/{total} hashed  ({elapsed:.1f}s)")

    # Save
    fieldnames = ["filename", "split", "aHash", "pHash", "hsvHash", "sHash"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_base = sum(1 for r in rows if r["split"] == "base")
    n_neg  = sum(1 for r in rows if r["split"] == "negative")
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
    print(f"  base: {n_base}  |  negative: {n_neg}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
