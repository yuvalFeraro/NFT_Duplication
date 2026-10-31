"""
step3b_hash_negatives.py
========================
Computes perceptual hashes for the 516 NEGATIVE reference images
(those in Data_sets/DISC-21/ that have NO query pair).

These images are NOT included in the 2409_data xlsx files, so we hash
them directly from the image files using imagehash (same library as
the rest of the project).

Output: data/hashes/negatives_hashes.csv  (516 rows)
Columns: filename, aHash, pHash, hsvHash, sHash

Run:
    python step3b_hash_negatives.py
"""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
import imagehash

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_V2   = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
V1        = Path(__file__).resolve().parents[1]
DISC21_DIR = BASE_V2 / "Data_sets" / "DISC-21"
OUT_CSV    = V1 / "data" / "hashes" / "negatives_hashes.csv"
SPLITS_DIR = V1 / "data" / "splits"

MAX_WORKERS = 6


def compute_hashes(img_path_str: str) -> dict | None:
    """Worker: load image and compute all 4 hashes."""
    try:
        p = Path(img_path_str)
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            return {
                "filename": p.name,
                "aHash":    str(imagehash.average_hash(rgb)),
                "pHash":    str(imagehash.phash(rgb)),
                "hsvHash":  str(imagehash.colorhash(rgb)),
                "sHash":    str(imagehash.crop_resistant_hash(rgb)),
            }
    except Exception as e:
        print(f"  ERROR hashing {img_path_str}: {e}")
        return None


def get_negative_filenames() -> list[str]:
    """Load negative filenames from all 4 split CSVs (train + test negatives)."""
    fns = []
    for split_file in ["train_negatives.csv", "test_negatives.csv"]:
        with open(SPLITS_DIR / split_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fns.append(row["filename"])
    return fns


def main():
    print("=" * 60)
    print("STEP 3b - Hash Negative Reference Images")
    print("=" * 60)

    neg_fns = get_negative_filenames()
    print(f"\n[1] Negatives to hash: {len(neg_fns)}")

    # Check all files exist
    missing = [fn for fn in neg_fns if not (DISC21_DIR / fn).exists()]
    if missing:
        print(f"  WARNING: {len(missing)} files not found in {DISC21_DIR}")
        for m in missing[:10]:
            print(f"    MISSING: {m}")
    
    paths = [str(DISC21_DIR / fn) for fn in neg_fns if (DISC21_DIR / fn).exists()]
    print(f"[2] Hashing {len(paths)} images with {MAX_WORKERS} workers...")

    results = {}
    completed = 0
    total = len(paths)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compute_hashes, p): p for p in paths}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results[res["filename"]] = res
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  {completed}/{total} hashed...")

    # Write CSV in the same order as neg_fns
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["filename", "aHash", "pHash", "hsvHash", "sHash"]
    rows_out = [results[fn] for fn in neg_fns if fn in results]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[3] Saved {len(rows_out)} rows -> {OUT_CSV}")
    print("\n[PASS]  Negatives hashing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
