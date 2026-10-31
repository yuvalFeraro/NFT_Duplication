"""
step4_train_and_test.py
=======================
Brute-force threshold search for the Minimal-Distance Detector on DISC-21.

Detector variants:
  k=1  - flag as duplicate if ANY hash distance < its threshold  (OR logic)
  k=2  - flag as duplicate if AT LEAST 2 hash distances < their thresholds
  k=3  - flag as duplicate if AT LEAST 3 hash distances < their thresholds

For each k, we sweep threshold combinations across all 4 hashes:
    aHash   : Hamming distance (0-64 bits)
    pHash   : Hamming distance (0-64 bits)
    hsvHash : Hamming distance (0-44 bits, 42-bit colorhash)
    sHash   : avg-block Hamming distance - hash_diff / num_segments

Objective: maximise Recall  subject to  FPR ≤ 0.20

Metrics reported:
  Train → Recall, FPR  (for every non-dominated combo)
  Test  → Recall, FPR, Precision, F1

Outputs (in results/):
    train_results_k1.csv
    train_results_k2.csv
    train_results_k3.csv
    test_report.txt

Run (steps 1-3 must be complete first):
    python step4_train_and_test.py
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any

import imagehash
from imagehash import ImageMultiHash
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
V1_DIR      = SCRIPT_DIR.parent
SPLITS_DIR  = V1_DIR / "data" / "splits"
HASHES_DIR  = V1_DIR / "data" / "hashes"
RESULTS_DIR = V1_DIR / "results"

# References live in:
BASE_V2        = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
DISC21_REFS_DIR    = BASE_V2 / "Data_sets" / "DISC-21"
DISC21_QUERIES_DIR = BASE_V2 / "dataset_duplications" / "DISC21_duplications"

MAX_FPR = 0.20          # FPR constraint during training
K_VALUES = [1, 2, 3]   # Detector variants to sweep

# Threshold grids (inclusive, step 1)
GRID_AHASH   = list(range(0, 22))   # 0,1,...,18  (19 values)
GRID_PHASH   = list(range(0, 22))   # 0,1,...,18  (19 values)
GRID_HSVHASH = list(range(0, 6))    # 0,1,...,5   ( 6 values)
GRID_SHASH   = list(range(0, 22))   # 0,1,...,18  (19 values)

HASH_NAMES  = ["aHash", "pHash", "hsvHash", "sHash"]
HASH_GRIDS  = [GRID_AHASH, GRID_PHASH, GRID_HSVHASH, GRID_SHASH]


# ---------------------------------------------------------------------------
# Hash parsing  (mirrors bk_tree.py / minimal_distance_detector.py)
# ---------------------------------------------------------------------------

def parse_hash(hash_name: str, hex_str: str) -> Any:
    """Parse a stored hex string back into the appropriate imagehash object."""
    hex_str = str(hex_str).strip()
    if not hex_str:
        return None
    if hash_name == "sHash":
        segments = [s.strip() for s in hex_str.split(",") if s.strip()]
        return ImageMultiHash([imagehash.hex_to_hash(s) for s in segments])
    elif hash_name == "hsvHash":
        # colorhash: 14 hue bins × 3 bits = 42 bits stored as 11-hex-char string
        total_bits = 42
        bit_string = bin(int(hex_str, 16))[2:].zfill(total_bits)
        arr = np.array([int(b) for b in bit_string]).reshape((14, 3))
        return imagehash.ImageHash(arr.astype(bool))
    else:
        return imagehash.hex_to_hash(hex_str)


def hash_distance(hash_name: str, h1: Any, h2: Any) -> float:
    """Compute the scalar Hamming distance between two parsed hashes."""
    if h1 is None or h2 is None:
        return float("inf")
    if hash_name == "sHash":
        # Same logic as sHash_hamming_distance in minimal_distance_detector.py:
        # hash_diff returns (matched_count, total_diff); we divide by n_segments of h1.
        n_segs = len(h1.segment_hashes)
        if n_segs == 0:
            return float("inf")
        _, total_diff = h1.hash_diff(h2, hamming_cutoff=100)
        return total_diff / n_segs
    else:
        return float(h1 - h2)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_hashes(csv_path: Path) -> dict[str, dict[str, Any]]:
    """
    Return {filename: {hash_name: parsed_hash_object, ...}}
    """
    records = load_csv(csv_path)
    store: dict[str, dict[str, Any]] = {}
    for row in records:
        fn = row["filename"].strip()
        store[fn] = {
            h: parse_hash(h, row.get(h, ""))
            for h in HASH_NAMES
        }
    return store


# ---------------------------------------------------------------------------
# Distance matrix builder
# ---------------------------------------------------------------------------

def build_distance_matrix(
    query_store: dict[str, dict[str, Any]],
    base_store:  dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Precompute distances:
        dist[query_fn][base_fn][hash_name] = float
    Returns a nested dict for fast lookup during threshold sweep.
    """
    dist: dict[str, dict[str, dict[str, float]]] = {}
    for qfn, q_hashes in query_store.items():
        dist[qfn] = {}
        for bfn, b_hashes in base_store.items():
            dist[qfn][bfn] = {
                h: hash_distance(h, q_hashes[h], b_hashes[h])
                for h in HASH_NAMES
            }
    return dist


def min_distances_to_base(
    query_fn: str,
    dist_matrix: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """
    For a single query, return the minimum distance to ANY base image per hash.
    min_d[hash_name] = min over all base images of dist[query_fn][base_fn][hash_name]
    """
    if query_fn not in dist_matrix:
        return {h: float("inf") for h in HASH_NAMES}
    per_base = dist_matrix[query_fn]
    mins: dict[str, float] = {}
    for h in HASH_NAMES:
        vals = [per_base[bfn][h] for bfn in per_base if not math.isinf(per_base[bfn][h])]
        mins[h] = min(vals) if vals else float("inf")
    return mins


# ---------------------------------------------------------------------------
# Detector logic
# ---------------------------------------------------------------------------

def predict(min_dists: dict[str, float], thresholds: dict[str, float], k: int) -> bool:
    """
    k=1: detected if ANY hash distance < threshold  (OR)
    k≥2: detected if AT LEAST k hashes have distance < threshold
    """
    hits = sum(1 for h in HASH_NAMES if min_dists[h] < thresholds[h])
    return hits >= k


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pos_min_dists: list[dict[str, float]],   # ground-truth duplicates
    neg_min_dists: list[dict[str, float]],   # ground-truth negatives
    thresholds: dict[str, float],
    k: int,
) -> dict[str, float]:
    """Compute TP, FP, FN, TN and derived metrics."""
    tp = sum(1 for md in pos_min_dists if predict(md, thresholds, k))
    fn = len(pos_min_dists) - tp
    fp = sum(1 for md in neg_min_dists if predict(md, thresholds, k))
    tn = len(neg_min_dists) - fp

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Recall": recall, "FPR": fpr, "Precision": precision, "F1": f1}


# ---------------------------------------------------------------------------
# Brute-force threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    pos_min_dists: list[dict[str, float]],
    neg_min_dists: list[dict[str, float]],
    k: int,
    results_csv: Path,
) -> dict:
    """
    Sweep all combinations of threshold grids for each hash.
    Record every configuration that satisfies FPR ≤ MAX_FPR.
    Return the best configuration (highest Recall among those satisfying constraint).
    """
    print(f"\n  [k={k}] Sweeping threshold combinations ...")
    total = math.prod(len(g) for g in HASH_GRIDS)
    print(f"         Total combinations: {total:,}")

    best = {"Recall": -1.0, "FPR": 1.0}
    best_thresholds: dict[str, float] = {}
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    checked = 0
    saved_rows = []

    for combo in itertools.product(*HASH_GRIDS):
        thresholds = dict(zip(HASH_NAMES, combo))
        metrics = evaluate(pos_min_dists, neg_min_dists, thresholds, k)
        checked += 1

        if metrics["FPR"] <= MAX_FPR:
            row = {**thresholds, **metrics}
            saved_rows.append(row)

            # Print progress for non-dominated improvements
            if metrics["Recall"] > best["Recall"] or (
                metrics["Recall"] == best["Recall"] and metrics["FPR"] < best["FPR"]
            ):
                best = metrics
                best_thresholds = thresholds.copy()
                elapsed = time.time() - t0
                print(
                    f"         [{elapsed:6.1f}s | {checked:>9,}/{total:,}]"
                    f"  k={k}"
                    f"  T_a={thresholds['aHash']:4.1f}"
                    f"  T_p={thresholds['pHash']:4.1f}"
                    f"  T_h={thresholds['hsvHash']:4.1f}"
                    f"  T_s={thresholds['sHash']:4.1f}"
                    f"  → Recall={metrics['Recall']:.4f}"
                    f"  FPR={metrics['FPR']:.4f}"
                )

    # Save all valid configs to CSV
    fieldnames = HASH_NAMES + ["TP", "FP", "FN", "TN", "Recall", "FPR", "Precision", "F1"]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by Recall desc, FPR asc
        saved_rows.sort(key=lambda r: (-r["Recall"], r["FPR"]))
        writer.writerows(saved_rows)

    elapsed_total = time.time() - t0
    print(f"         Sweep done in {elapsed_total:.1f}s. "
          f"Valid configs (FPR≤{MAX_FPR}): {len(saved_rows):,}. "
          f"Saved to {results_csv.name}")

    if not best_thresholds:
        print(f"  WARNING: no configuration satisfies FPR ≤ {MAX_FPR} for k={k}!")
    return best_thresholds, best


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 4 - Brute-Force Threshold Search + Test Evaluation")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load split manifests ----
    print("\n[1] Loading split manifests ...")
    train_base_rows   = load_csv(SPLITS_DIR / "train_base.csv")
    test_base_rows    = load_csv(SPLITS_DIR / "test_base.csv")
    train_neg_rows    = load_csv(SPLITS_DIR / "train_negatives.csv")
    test_neg_rows     = load_csv(SPLITS_DIR / "test_negatives.csv")
    train_dup_rows    = load_csv(SPLITS_DIR / "train_duplications.csv")
    test_dup_rows     = load_csv(SPLITS_DIR / "test_duplications.csv")

    print(f"    Train: {len(train_base_rows)} base | {len(train_neg_rows)} negatives | {len(train_dup_rows)} duplications")
    print(f"    Test : {len(test_base_rows)} base  | {len(test_neg_rows)} negatives  | {len(test_dup_rows)} duplications")

    # ---- Load pre-computed hashes ----
    print("\n[2] Loading pre-computed hashes ...")
    ref_store   = load_hashes(HASHES_DIR / "references_hashes.csv")
    query_store = load_hashes(HASHES_DIR / "queries_hashes.csv")
    neg_hash_path = HASHES_DIR / "negatives_hashes.csv"
    if not neg_hash_path.exists():
        print("\nERROR: negatives_hashes.csv not found.")
        print("Please run step3b_hash_negatives.py first, then re-run this script.")
        sys.exit(1)
    neg_store   = load_hashes(neg_hash_path)
    print(f"    References hashed : {len(ref_store)}")
    print(f"    Queries hashed    : {len(query_store)}")
    print(f"    Negatives hashed  : {len(neg_store)}")

    # ---- Build hash sub-stores per split ----
    train_base_fns = set(r["filename"] for r in train_base_rows)
    test_base_fns  = set(r["filename"] for r in test_base_rows)
    train_dup_fns  = [r["filename"] for r in train_dup_rows]
    test_dup_fns   = [r["filename"] for r in test_dup_rows]
    train_neg_fns  = [r["filename"] for r in train_neg_rows]
    test_neg_fns   = [r["filename"] for r in test_neg_rows]

    train_base_store = {fn: ref_store[fn] for fn in train_base_fns if fn in ref_store}
    test_base_store  = {fn: ref_store[fn] for fn in test_base_fns  if fn in ref_store}
    train_neg_store  = {fn: neg_store[fn] for fn in train_neg_fns  if fn in neg_store}
    test_neg_store   = {fn: neg_store[fn] for fn in test_neg_fns   if fn in neg_store}
    train_dup_store  = {fn: query_store[fn] for fn in train_dup_fns if fn in query_store}
    test_dup_store   = {fn: query_store[fn] for fn in test_dup_fns  if fn in query_store}

    # ---- Precompute distance matrices ----
    print("\n[3] Precomputing distance matrices ...")
    t0 = time.time()
    print("    Train: duplications → base ...")
    train_dup_vs_base = build_distance_matrix(train_dup_store, train_base_store)
    print("    Train: negatives → base ...")
    train_neg_vs_base = build_distance_matrix(train_neg_store, train_base_store)
    print("    Test:  duplications → base ...")
    test_dup_vs_base  = build_distance_matrix(test_dup_store,  test_base_store)
    print("    Test:  negatives → base ...")
    test_neg_vs_base  = build_distance_matrix(test_neg_store,  test_base_store)
    print(f"    Distance matrices built in {time.time()-t0:.1f}s")

    # ---- Min-distances per image (for fast sweep) ----
    print("\n[4] Computing per-image min-distances to base ...")
    train_pos_min = [min_distances_to_base(fn, train_dup_vs_base) for fn in train_dup_fns]
    train_neg_min = [min_distances_to_base(fn, train_neg_vs_base) for fn in train_neg_fns]
    test_pos_min  = [min_distances_to_base(fn, test_dup_vs_base)  for fn in test_dup_fns]
    test_neg_min  = [min_distances_to_base(fn, test_neg_vs_base)  for fn in test_neg_fns]

    # ---- Threshold sweep for each k ----
    print("\n" + "=" * 70)
    print("TRAINING PHASE - Brute-force threshold sweep")
    print(f"Objective: Maximize Recall  subject to  FPR ≤ {MAX_FPR}")
    print("=" * 70)

    best_results: dict[int, tuple[dict, dict]] = {}   # k → (best_thresholds, best_metrics)

    for k in K_VALUES:
        csv_path = RESULTS_DIR / f"train_results_k{k}.csv"
        best_thresh, best_met = sweep_thresholds(
            train_pos_min, train_neg_min, k, csv_path
        )
        best_results[k] = (best_thresh, best_met)

    # ---- Training summary ----
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'k':>3}  {'T_aHash':>8}  {'T_pHash':>8}  {'T_hsvHash':>10}  {'T_sHash':>8}"
          f"  {'Recall':>8}  {'FPR':>8}")
    for k in K_VALUES:
        bt, bm = best_results[k]
        if not bt:
            print(f"  k={k}  - no valid config found")
            continue
        print(f"  k={k}"
              f"  {bt.get('aHash',0):>8.1f}  {bt.get('pHash',0):>8.1f}"
              f"  {bt.get('hsvHash',0):>10.1f}  {bt.get('sHash',0):>8.1f}"
              f"  {bm['Recall']:>8.4f}  {bm['FPR']:>8.4f}")

    # ---- Test phase ----
    print("\n" + "=" * 70)
    print("TEST PHASE")
    print("=" * 70)

    report_lines: list[str] = ["NFT_program_V1 - Test Phase Report", "=" * 50, ""]
    print(f"\n{'k':>3}  {'Recall':>8}  {'FPR':>8}  {'Precision':>10}  {'F1':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")

    for k in K_VALUES:
        bt, _ = best_results[k]
        if not bt:
            line = f"k={k}: no valid training configuration found."
            print("  " + line)
            report_lines.append(line)
            continue
        test_metrics = evaluate(test_pos_min, test_neg_min, bt, k)
        line = (
            f"k={k}"
            f"  Recall={test_metrics['Recall']:.4f}"
            f"  FPR={test_metrics['FPR']:.4f}"
            f"  Precision={test_metrics['Precision']:.4f}"
            f"  F1={test_metrics['F1']:.4f}"
            f"  TP={test_metrics['TP']}  FP={test_metrics['FP']}"
            f"  FN={test_metrics['FN']}  TN={test_metrics['TN']}"
        )
        print("  " + line)
        report_lines += [
            f"Detector k={k}",
            f"  Best thresholds : aHash={bt['aHash']}  pHash={bt['pHash']}"
            f"  hsvHash={bt['hsvHash']}  sHash={bt['sHash']}",
            f"  TP={test_metrics['TP']}  FP={test_metrics['FP']}"
            f"  FN={test_metrics['FN']}  TN={test_metrics['TN']}",
            f"  Recall    = {test_metrics['Recall']:.4f}",
            f"  FPR       = {test_metrics['FPR']:.4f}",
            f"  Precision = {test_metrics['Precision']:.4f}",
            f"  F1        = {test_metrics['F1']:.4f}",
            "",
        ]

    # Save report
    report_path = RESULTS_DIR / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[PASS]  Test report saved to {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
