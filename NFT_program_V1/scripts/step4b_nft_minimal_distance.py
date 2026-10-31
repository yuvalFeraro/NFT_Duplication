"""
step4b_nft_minimal_distance.py
==============================
Applies the same Minimal-Distance Detector grid search as step4_train_and_test.py,
but on the NFT_program_V2 datasets (Azuki, Bored Ape Yacht Club, Lil Pudgys).

Data source: NFT_program_V2/data_manifests/
  train_base.csv       - 640 original images (base tree) - hashes pre-computed
  train_fakes.csv      - 562 fake images (positive queries) with ref_image column
  train_originals.csv  - 560 original images (negatives, not in base)
  test_base.csv        - 160 original images (test base tree)
  test_fakes.csv       - 138 fake images (test positives)
  test_originals.csv   - 140 original images (test negatives)

Detector variants (same as step4):
  k=1  flag as duplicate if ANY hash distance < threshold  (OR logic)
  k=2  flag if AT LEAST 2 hash distances < their thresholds
  k=3  flag if AT LEAST 3 hash distances < their thresholds

Evaluation: image-level (naturally, since we pre-reduce to min-distances)
  FP = negative image where the threshold condition fires
  TP = fake query where the threshold condition fires

Objective: Maximize Recall  subject to  FPR <= 0.20

Threshold grids (same as step4):
  aHash   : 0..21  (22 values)
  pHash   : 0..21  (22 values)
  hsvHash : 0..5   ( 6 values)
  sHash   : 0..21  (22 values)
  Total combinations: 22 x 22 x 6 x 22 = 64,152

Outputs (in NFT_program_V1/results/):
  train_results_nft_k1.csv
  train_results_nft_k2.csv
  train_results_nft_k3.csv
  test_report_nft.txt

Run:
  python step4b_nft_minimal_distance.py
"""

from __future__ import annotations

import csv
import itertools
import math
import random
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
RESULTS_DIR = V1_DIR / "results"

BASE_V2         = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
MANIFESTS_DIR   = BASE_V2 / "data_manifests"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HASH_NAMES  = ["aHash", "pHash", "hsvHash", "sHash"]

MAX_FPR  = 0.20
K_VALUES = [1, 2, 3]

# Threshold grids (inclusive, step 1) — same as step4_train_and_test.py
GRID_AHASH   = list(range(0, 22))   # 22 values
GRID_PHASH   = list(range(0, 22))   # 22 values
GRID_HSVHASH = list(range(0, 6))    #  6 values
GRID_SHASH   = list(range(0, 22))   # 22 values
HASH_GRIDS   = [GRID_AHASH, GRID_PHASH, GRID_HSVHASH, GRID_SHASH]

# ---------------------------------------------------------------------------
# Subsampling — match DISC-21 quantities (set to None to use all data)
# ---------------------------------------------------------------------------
RANDOM_SEED    = 42
TRAIN_N_FAKES  = 413   # same as DISC-21 train split
TRAIN_N_NEGS   = 413
TEST_N_FAKES   = 103   # same as DISC-21 test split
TEST_N_NEGS    = 103


# ---------------------------------------------------------------------------
# Hash parsing  (identical to step4_train_and_test.py)
# ---------------------------------------------------------------------------

def parse_hash(hash_name: str, hex_str: str) -> Any:
    """Convert a stored hex string back into the appropriate imagehash object."""
    hex_str = str(hex_str).strip()
    if not hex_str or hex_str.lower() == "nan":
        return None
    if hash_name == "sHash":
        segments = [s.strip() for s in hex_str.split(",") if s.strip()]
        return ImageMultiHash([imagehash.hex_to_hash(s) for s in segments])
    elif hash_name == "hsvHash":
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
        n_segs = len(h1.segment_hashes)
        if n_segs == 0:
            return float("inf")
        _, total_diff = h1.hash_diff(h2, hamming_cutoff=100)
        return total_diff / n_segs
    else:
        return float(h1 - h2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def subsample(rows: list[dict], n: int | None, rng: random.Random) -> list[dict]:
    """Return a random subset of `n` rows (without replacement). If n is None or
    n >= len(rows), return all rows unchanged."""
    if n is None or n >= len(rows):
        return rows
    return rng.sample(rows, n)


def build_hash_store(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """
    Parse hash hex strings from manifest rows into imagehash objects.
    Returns {filename: {hash_name: hash_object}}.
    """
    store: dict[str, dict[str, Any]] = {}
    for row in rows:
        fn = row["filename"].strip()
        store[fn] = {h: parse_hash(h, row.get(h, "")) for h in HASH_NAMES}
    return store


# ---------------------------------------------------------------------------
# Full pairwise min-distance  (same logic as step4_train_and_test.py)
# ---------------------------------------------------------------------------

def compute_min_distances(
    query_fns: list[str],
    query_store: dict[str, dict[str, Any]],
    base_store: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    For each query, compute its minimum Hamming distance to ANY base image
    per hash. Returns {query_fn: {hash_name: min_distance}}.
    """
    result: dict[str, dict[str, float]] = {}
    total = len(query_fns)
    for i, qfn in enumerate(query_fns):
        if qfn not in query_store:
            continue
        q_hashes = query_store[qfn]
        min_d = {h: float("inf") for h in HASH_NAMES}

        for b_hashes in base_store.values():
            for h in HASH_NAMES:
                d = hash_distance(h, q_hashes[h], b_hashes[h])
                if d < min_d[h]:
                    min_d[h] = d

        result[qfn] = min_d

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"    {i+1}/{total} done ...")
    return result


# ---------------------------------------------------------------------------
# Detector logic  (identical to step4_train_and_test.py)
# ---------------------------------------------------------------------------

def predict_image(min_dists: dict[str, float], thresholds: dict[str, float], k: int) -> bool:
    """k=1: OR; k>=2: at least k hashes vote."""
    hits = sum(1 for h in HASH_NAMES if min_dists[h] < thresholds[h])
    return hits >= k


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def sweep_thresholds(
    pos_min_dists: list[dict[str, float]],  # fake queries
    neg_min_dists: list[dict[str, float]],  # negatives
    k: int,
    results_csv: Path,
) -> tuple[dict, dict]:
    """
    Sweep all threshold combinations. Record every config with FPR <= MAX_FPR.
    Return (best_thresholds, best_metrics).
    """
    print(f"\n  [k={k}] Sweeping threshold combinations ...")
    total = math.prod(len(g) for g in HASH_GRIDS)
    print(f"         Total combinations: {total:,}")

    best: dict = {"Recall": -1.0, "FPR": 1.0}
    best_thresholds: dict[str, float] = {}
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    # Pre-extract arrays for speed
    f_a = np.array([d["aHash"]   for d in pos_min_dists])
    f_p = np.array([d["pHash"]   for d in pos_min_dists])
    f_h = np.array([d["hsvHash"] for d in pos_min_dists])
    f_s = np.array([d["sHash"]   for d in pos_min_dists])

    n_a = np.array([d["aHash"]   for d in neg_min_dists])
    n_p = np.array([d["pHash"]   for d in neg_min_dists])
    n_h = np.array([d["hsvHash"] for d in neg_min_dists])
    n_s = np.array([d["sHash"]   for d in neg_min_dists])

    len_f = len(f_a)
    len_n = len(n_a)

    t0 = time.time()
    saved_rows: list[dict] = []

    for combo in itertools.product(*HASH_GRIDS):
        a_t, p_t, h_t, s_t = combo
        thresholds = dict(zip(HASH_NAMES, combo))

        if k == 1:
            tp_mask = (f_a < a_t) | (f_p < p_t) | (f_h < h_t) | (f_s < s_t)
            fp_mask = (n_a < a_t) | (n_p < p_t) | (n_h < h_t) | (n_s < s_t)
        else:
            f_votes = (f_a < a_t).astype(int) + (f_p < p_t).astype(int) + \
                      (f_h < h_t).astype(int) + (f_s < s_t).astype(int)
            n_votes = (n_a < a_t).astype(int) + (n_p < p_t).astype(int) + \
                      (n_h < h_t).astype(int) + (n_s < s_t).astype(int)
            tp_mask = f_votes >= k
            fp_mask = n_votes >= k

        tp = int(np.sum(tp_mask))
        fp = int(np.sum(fp_mask))
        fn = len_f - tp
        tn = len_n - fp

        recall = tp / len_f if len_f > 0 else 0.0
        fpr    = fp / len_n if len_n > 0 else 0.0

        if fpr <= MAX_FPR:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            row = {**thresholds, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "Recall": recall, "FPR": fpr, "Precision": precision, "F1": f1}
            saved_rows.append(row)

            if recall > best["Recall"] or (
                recall == best["Recall"] and fpr < best["FPR"]
            ):
                best = row
                best_thresholds = thresholds.copy()
                elapsed = time.time() - t0
                print(
                    f"         [{elapsed:6.1f}s]"
                    f"  k={k}"
                    f"  T_a={a_t:3}  T_p={p_t:3}  T_h={h_t:3}  T_s={s_t:3}"
                    f"  -> Recall={recall:.4f}  FPR={fpr:.4f}"
                )

    # Save all valid configs, sorted by Recall desc
    fieldnames = HASH_NAMES + ["TP", "FP", "FN", "TN", "Recall", "FPR", "Precision", "F1"]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        saved_rows.sort(key=lambda r: (-r["Recall"], r["FPR"]))
        writer.writerows(saved_rows)

    elapsed_total = time.time() - t0
    print(f"         Sweep done in {elapsed_total:.1f}s. "
          f"Valid configs (FPR<={MAX_FPR}): {len(saved_rows):,}. "
          f"Saved to {results_csv.name}")

    if not best_thresholds:
        print(f"  WARNING: no configuration satisfies FPR <= {MAX_FPR} for k={k}!")
    return best_thresholds, best


# ---------------------------------------------------------------------------
# Test evaluation  (image-level, naturally)
# ---------------------------------------------------------------------------

def evaluate_test(
    pos_min_dists: list[dict[str, float]],
    neg_min_dists: list[dict[str, float]],
    thresholds: dict[str, float],
    k: int,
) -> dict:
    tp = sum(1 for d in pos_min_dists if predict_image(d, thresholds, k))
    fp = sum(1 for d in neg_min_dists if predict_image(d, thresholds, k))
    fn = len(pos_min_dists) - tp
    tn = len(neg_min_dists) - fp

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Recall": recall, "FPR": fpr, "Precision": precision, "F1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 4b — Minimal Distance Detector: NFT V2 Datasets")
    print("(Azuki + Bored Ape Yacht Club + Lil Pudgys)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # ---- Load manifests ----
    print("\n[1] Loading manifests ...")
    all_train_fakes = load_csv(MANIFESTS_DIR / "train_fakes.csv")
    all_test_fakes  = load_csv(MANIFESTS_DIR / "test_fakes.csv")
    all_train_negs  = load_csv(MANIFESTS_DIR / "train_originals.csv")
    all_test_negs   = load_csv(MANIFESTS_DIR / "test_originals.csv")
    all_train_base  = load_csv(MANIFESTS_DIR / "train_base.csv")
    all_test_base   = load_csv(MANIFESTS_DIR / "test_base.csv")

    print(f"  Full data — Train: {len(all_train_base)} base | "
          f"{len(all_train_fakes)} fakes | {len(all_train_negs)} negatives")
    print(f"  Full data — Test : {len(all_test_base)} base  | "
          f"{len(all_test_fakes)} fakes  | {len(all_test_negs)} negatives")

    # ---- Subsample fakes (with seed) ----
    train_fakes_rows = subsample(all_train_fakes, TRAIN_N_FAKES, rng)
    test_fakes_rows  = subsample(all_test_fakes,  TEST_N_FAKES,  rng)

    # ---- Derive base from the ref_images of the sampled fakes ----
    # (mirrors DISC-21 structure: base = one ref per fake query)
    train_base_fns = {r["ref_image"].strip() for r in train_fakes_rows}
    test_base_fns  = {r["ref_image"].strip() for r in test_fakes_rows}
    train_base_lookup = {r["filename"].strip(): r for r in all_train_base}
    test_base_lookup  = {r["filename"].strip(): r for r in all_test_base}
    train_base_rows = [train_base_lookup[fn] for fn in train_base_fns
                       if fn in train_base_lookup]
    test_base_rows  = [test_base_lookup[fn]  for fn in test_base_fns
                       if fn in test_base_lookup]

    # ---- Subsample negatives ----
    train_neg_rows = subsample(all_train_negs, TRAIN_N_NEGS, rng)
    test_neg_rows  = subsample(all_test_negs,  TEST_N_NEGS,  rng)

    print(f"\n  Subsampled — Train: {len(train_base_rows)} base | "
          f"{len(train_fakes_rows)} fakes | {len(train_neg_rows)} negatives")
    print(f"  Subsampled — Test : {len(test_base_rows)} base  | "
          f"{len(test_fakes_rows)} fakes  | {len(test_neg_rows)} negatives")

    # ---- Build hash stores ----
    print("\n[2] Parsing hashes ...")
    train_base_store = build_hash_store(train_base_rows)
    test_base_store  = build_hash_store(test_base_rows)
    train_fake_store = build_hash_store(train_fakes_rows)
    test_fake_store  = build_hash_store(test_fakes_rows)
    train_neg_store  = build_hash_store(train_neg_rows)
    test_neg_store   = build_hash_store(test_neg_rows)
    print(f"  Hash stores built.")

    # ---- Compute min-distances ----
    print("\n[3] Computing pairwise min-distances ...")

    print("  Train fakes -> train base ...")
    t0 = time.time()
    train_fake_mins = compute_min_distances(
        [r["filename"] for r in train_fakes_rows],
        train_fake_store, train_base_store,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  Train negatives -> train base ...")
    t0 = time.time()
    train_neg_mins = compute_min_distances(
        [r["filename"] for r in train_neg_rows],
        train_neg_store, train_base_store,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  Test fakes -> test base ...")
    t0 = time.time()
    test_fake_mins = compute_min_distances(
        [r["filename"] for r in test_fakes_rows],
        test_fake_store, test_base_store,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  Test negatives -> test base ...")
    t0 = time.time()
    test_neg_mins = compute_min_distances(
        [r["filename"] for r in test_neg_rows],
        test_neg_store, test_base_store,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Convert to flat lists for the sweep
    train_pos_dists = list(train_fake_mins.values())
    train_neg_dists = list(train_neg_mins.values())
    test_pos_dists  = list(test_fake_mins.values())
    test_neg_dists  = list(test_neg_mins.values())

    print(f"\n  Train positives with valid distances: {len(train_pos_dists)}/{len(train_fakes_rows)}")
    print(f"  Train negatives with valid distances: {len(train_neg_dists)}/{len(train_neg_rows)}")
    print(f"  Test  positives with valid distances: {len(test_pos_dists)}/{len(test_fakes_rows)}")
    print(f"  Test  negatives with valid distances: {len(test_neg_dists)}/{len(test_neg_rows)}")

    # ---- Grid search (training phase) ----
    print("\n" + "=" * 70)
    print("TRAINING PHASE — Brute-force threshold sweep")
    print(f"Objective: Maximize Recall  subject to  FPR <= {MAX_FPR}")
    print("=" * 70)

    best_results: dict[int, tuple[dict, dict]] = {}

    for k in K_VALUES:
        csv_path = RESULTS_DIR / f"train_results_nft_k{k}.csv"
        best_thresh, best_met = sweep_thresholds(
            train_pos_dists, train_neg_dists, k, csv_path
        )
        best_results[k] = (best_thresh, best_met)

    # ---- Training summary ----
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'k':>3}  {'T_aHash':>8}  {'T_pHash':>8}  {'T_hsvHash':>10}  "
          f"{'T_sHash':>8}  {'Recall':>8}  {'FPR':>8}")
    for k in K_VALUES:
        bt, bm = best_results[k]
        if not bt:
            print(f"  k={k}  - no valid config found")
            continue
        print(f"  k={k}"
              f"  {bt.get('aHash',0):>8}  {bt.get('pHash',0):>8}"
              f"  {bt.get('hsvHash',0):>10}  {bt.get('sHash',0):>8}"
              f"  {bm['Recall']:>8.4f}  {bm['FPR']:>8.4f}")

    # ---- Test phase ----
    print("\n" + "=" * 70)
    print("TEST PHASE  (image-level evaluation)")
    print("=" * 70)
    print(f"  Positive test images : {len(test_pos_dists)}")
    print(f"  Negative test images : {len(test_neg_dists)}")

    report_lines: list[str] = [
        "NFT_program_V2 Datasets — Minimal Distance Detector Test Report\n",
        "=" * 60 + "\n",
        f"Datasets    : Azuki + Bored Ape Yacht Club + Lil Pudgys\n",
        f"Subsampled to match DISC-21 quantities (seed={RANDOM_SEED})\n",
        f"Train base  : {len(train_base_rows)} images\n",
        f"Train fakes : {len(train_pos_dists)} images\n",
        f"Train neg   : {len(train_neg_dists)} images\n",
        f"Test base   : {len(test_base_rows)} images\n",
        f"Test fakes  : {len(test_pos_dists)} images\n",
        f"Test neg    : {len(test_neg_dists)} images\n",
        "=" * 60 + "\n\n",
    ]

    print(f"\n{'k':>3}  {'Recall':>8}  {'FPR':>8}  {'Precision':>10}  "
          f"{'F1':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")

    for k in K_VALUES:
        bt, _ = best_results[k]
        if not bt:
            line = f"k={k}: no valid training configuration found.\n"
            print("  " + line)
            report_lines.append(line)
            continue

        m = evaluate_test(test_pos_dists, test_neg_dists, bt, k)
        print(f"  k={k}"
              f"  {m['Recall']:>8.4f}  {m['FPR']:>8.4f}"
              f"  {m['Precision']:>10.4f}  {m['F1']:>8.4f}"
              f"  {m['TP']:>5}  {m['FP']:>5}  {m['FN']:>5}  {m['TN']:>5}")

        report_lines += [
            f"Detector k={k}\n",
            f"  Best thresholds : aHash={bt['aHash']}  pHash={bt['pHash']}"
            f"  hsvHash={bt['hsvHash']}  sHash={bt['sHash']}\n",
            f"  Train Recall={best_results[k][1]['Recall']:.4f}"
            f"  Train FPR={best_results[k][1]['FPR']:.4f}\n",
            f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}\n",
            f"  Recall    = {m['Recall']:.4f}\n",
            f"  FPR       = {m['FPR']:.4f}\n",
            f"  Precision = {m['Precision']:.4f}\n",
            f"  F1        = {m['F1']:.4f}\n\n",
        ]

    report_path = RESULTS_DIR / "test_report_nft.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n[PASS]  Report saved to {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
