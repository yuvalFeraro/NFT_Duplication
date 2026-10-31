"""
cp_step2_detector.py
====================
Minimal Distance Detector for the CryptoPunk collection.

Data:
  Base tree   = CryptoPunk originals (200) + 200 imgs from imgs/
  Positives   = CryptoPunk manipulations (202)  [ref parsed from filename]
  Negatives   = 200 imgs from imgs/ (different from the 200 in base)

Hashes:
  CryptoPunk originals + manipulations  -> final_test_metadata.xlsx (pre-computed)
  imgs images                           -> cryptoPunk/data/imgs_hashes.csv  (step1)

Split (80/20 by CryptoPunk original):
  Train: 160 CP originals + 160 imgs_base | ~162 positives | 160 negatives
  Test :  40 CP originals +  40 imgs_base |  ~40 positives |  40 negatives

Detector: same as step4 / step4b
  k=1 : OR logic (any hash fires)
  k=2 : at least 2 hashes fire
  k=3 : at least 3 hashes fire
  Max FPR: 0.20 during training

Outputs (in NFT_program_V1/cryptoPunk/results/):
  train_results_cp_k1.csv
  train_results_cp_k2.csv
  train_results_cp_k3.csv
  test_report_cp.txt

Run:
    python cp_step2_detector.py
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import imagehash
from imagehash import ImageMultiHash
import numpy as np
import openpyxl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
CP_DATA_DIR  = SCRIPT_DIR / "data"
RESULTS_DIR  = SCRIPT_DIR / "results"

CP_RAW_DIR   = Path(r"C:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets\CryptoPunk_data")
ORIG_DIR     = CP_RAW_DIR / "original"
MANIP_DIR    = CP_RAW_DIR / "mamipulations"
XLSX_PATH    = CP_RAW_DIR / "final_test_metadata.xlsx"
IMGS_CSV     = CP_DATA_DIR / "imgs_hashes.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HASH_NAMES  = ["aHash", "pHash", "hsvHash", "sHash"]
RANDOM_SEED = 42
TRAIN_RATIO = 0.80       # 80% of CP originals -> train

MAX_FPR  = 0.20
K_VALUES = [1, 2, 3]

GRID_AHASH   = list(range(0, 22))
GRID_PHASH   = list(range(0, 22))
GRID_HSVHASH = list(range(0, 6))
GRID_SHASH   = list(range(0, 22))
HASH_GRIDS   = [GRID_AHASH, GRID_PHASH, GRID_HSVHASH, GRID_SHASH]


# ---------------------------------------------------------------------------
# Hash parsing  (identical to step4 / step4b)
# ---------------------------------------------------------------------------

def parse_hash(hash_name: str, hex_str: str) -> Any:
    hex_str = str(hex_str).strip()
    if not hex_str or hex_str.lower() in ("nan", "none", ""):
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

def load_xlsx_hashes() -> tuple[dict, dict]:
    """
    Load CryptoPunk hashes from the xlsx.
    Returns:
      orig_hashes  : {filename: {hash_name: parsed_hash}}
      manip_hashes : {filename: {hash_name: parsed_hash, 'ref_image': str}}
    """
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    idx = {h: i+1 for i, h in enumerate(headers)}

    orig_hashes:  dict[str, dict] = {}
    manip_hashes: dict[str, dict] = {}

    for row in ws.iter_rows(min_row=2, values_only=True):
        def get(col): return row[idx[col]-1]

        orig_fn  = str(get("original_image")).strip()
        copy_fn  = str(get("copy_image")).strip()
        is_copy  = int(get("is_copy") or 0)

        # Originals
        if orig_fn not in orig_hashes:
            orig_hashes[orig_fn] = {
                "aHash":   parse_hash("aHash",   str(get("aHash_original")   or "")),
                "pHash":   parse_hash("pHash",   str(get("pHash_original")   or "")),
                "hsvHash": parse_hash("hsvHash", str(get("hsvHash_original") or "")),
                "sHash":   parse_hash("sHash",   str(get("sHash_original")   or "")),
            }

        # Manipulations: only grab the row where this copy IS a copy of this original
        if is_copy == 1 and copy_fn not in manip_hashes:
            manip_hashes[copy_fn] = {
                "aHash":     parse_hash("aHash",   str(get("aHash_copy")   or "")),
                "pHash":     parse_hash("pHash",   str(get("pHash_copy")   or "")),
                "hsvHash":   parse_hash("hsvHash", str(get("hsvHash_copy") or "")),
                "sHash":     parse_hash("sHash",   str(get("sHash_copy")   or "")),
                "ref_image": orig_fn,
            }

    wb.close()
    return orig_hashes, manip_hashes


def load_imgs_hashes() -> tuple[dict, dict]:
    """
    Load pre-computed imgs hashes from csv.
    Returns (imgs_base_store, imgs_neg_store).
    """
    if not IMGS_CSV.exists():
        print(f"ERROR: {IMGS_CSV} not found. Run cp_step1_hash_imgs.py first.")
        sys.exit(1)
    base: dict[str, dict] = {}
    neg:  dict[str, dict] = {}
    with open(IMGS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn    = row["filename"].strip()
            split = row["split"].strip()
            parsed = {h: parse_hash(h, row[h]) for h in HASH_NAMES}
            if split == "base":
                base[fn] = parsed
            else:
                neg[fn] = parsed
    return base, neg


# ---------------------------------------------------------------------------
# Min-distance computation
# ---------------------------------------------------------------------------

def compute_min_distances(
    query_fns: list[str],
    query_store: dict[str, dict],
    base_store:  dict[str, dict],
) -> dict[str, dict[str, float]]:
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
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"    {i+1}/{total} done ...")
    return result


# ---------------------------------------------------------------------------
# Detector & grid search
# ---------------------------------------------------------------------------

def predict_image(min_dists: dict[str, float], thresholds: dict[str, float], k: int) -> bool:
    hits = sum(1 for h in HASH_NAMES if min_dists[h] < thresholds[h])
    return hits >= k


def sweep_thresholds(
    pos_min_dists: list[dict[str, float]],
    neg_min_dists: list[dict[str, float]],
    k: int,
    results_csv: Path,
) -> tuple[dict, dict]:
    print(f"\n  [k={k}] Sweeping {math.prod(len(g) for g in HASH_GRIDS):,} combinations ...")
    best: dict = {"Recall": -1.0, "FPR": 1.0}
    best_thresholds: dict[str, float] = {}
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    f_a = np.array([d["aHash"]   for d in pos_min_dists])
    f_p = np.array([d["pHash"]   for d in pos_min_dists])
    f_h = np.array([d["hsvHash"] for d in pos_min_dists])
    f_s = np.array([d["sHash"]   for d in pos_min_dists])
    n_a = np.array([d["aHash"]   for d in neg_min_dists])
    n_p = np.array([d["pHash"]   for d in neg_min_dists])
    n_h = np.array([d["hsvHash"] for d in neg_min_dists])
    n_s = np.array([d["sHash"]   for d in neg_min_dists])
    len_f = len(f_a); len_n = len(n_a)

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

        tp = int(np.sum(tp_mask)); fp = int(np.sum(fp_mask))
        fn = len_f - tp; tn = len_n - fp
        recall = tp / len_f if len_f > 0 else 0.0
        fpr    = fp / len_n if len_n > 0 else 0.0

        if fpr <= MAX_FPR:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            row = {**thresholds, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "Recall": recall, "FPR": fpr, "Precision": precision, "F1": f1}
            saved_rows.append(row)
            if recall > best["Recall"] or (recall == best["Recall"] and fpr < best["FPR"]):
                best = row; best_thresholds = thresholds.copy()
                print(f"         [{time.time()-t0:6.1f}s]  k={k}"
                      f"  T_a={a_t:3}  T_p={p_t:3}  T_h={h_t:3}  T_s={s_t:3}"
                      f"  -> Recall={recall:.4f}  FPR={fpr:.4f}")

    fieldnames = HASH_NAMES + ["TP", "FP", "FN", "TN", "Recall", "FPR", "Precision", "F1"]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        saved_rows.sort(key=lambda r: (-r["Recall"], r["FPR"]))
        writer.writerows(saved_rows)
    print(f"         Sweep done in {time.time()-t0:.1f}s. "
          f"Valid configs: {len(saved_rows):,}. Saved -> {results_csv.name}")
    if not best_thresholds:
        print(f"  WARNING: no valid config for k={k}!")
    return best_thresholds, best


def evaluate_test(pos_dists, neg_dists, thresholds, k) -> dict:
    tp = sum(1 for d in pos_dists if predict_image(d, thresholds, k))
    fp = sum(1 for d in neg_dists if predict_image(d, thresholds, k))
    fn = len(pos_dists) - tp; tn = len(neg_dists) - fp
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Recall": recall, "FPR": fpr, "Precision": precision, "F1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    print("=" * 70)
    print("CP STEP 2 — Minimal Distance Detector: CryptoPunk")
    print("=" * 70)

    # ---- Load hashes ----
    print("\n[1] Loading CryptoPunk hashes from xlsx ...")
    orig_hashes, manip_hashes = load_xlsx_hashes()
    print(f"  Originals loaded   : {len(orig_hashes)}")
    print(f"  Manipulations loaded: {len(manip_hashes)}")

    print("\n[2] Loading imgs hashes from CSV ...")
    imgs_base_store, imgs_neg_store = load_imgs_hashes()
    print(f"  imgs base    : {len(imgs_base_store)}")
    print(f"  imgs negative: {len(imgs_neg_store)}")

    # ---- Parse ref_image from manipulation filename ----
    # e.g. cryptopunk_0_crop_65555321384..png -> cryptopunk_0.png
    # Use the ref_image already stored from xlsx (where is_copy=1)
    # Verify parse vs xlsx
    _re = re.compile(r"^cryptopunk_(\d+)_")
    manip_list = []   # list of (manip_fn, ref_fn)
    for fn, info in manip_hashes.items():
        ref_fn = info.get("ref_image", "")
        if not ref_fn:
            m = _re.match(fn)
            ref_fn = f"cryptopunk_{m.group(1)}.png" if m else ""
        if ref_fn in orig_hashes:
            manip_list.append((fn, ref_fn))
        else:
            print(f"  WARNING: ref '{ref_fn}' for {fn} not found in originals, skipping")

    print(f"\n  Manipulations with valid reference: {len(manip_list)}")

    # ---- Split by CryptoPunk original (80/20) ----
    all_orig_fns = sorted(orig_hashes.keys())
    rng.shuffle(all_orig_fns)
    n_train_orig = int(len(all_orig_fns) * TRAIN_RATIO)
    train_orig_fns = set(all_orig_fns[:n_train_orig])
    test_orig_fns  = set(all_orig_fns[n_train_orig:])

    train_manips = [(fn, ref) for fn, ref in manip_list if ref in train_orig_fns]
    test_manips  = [(fn, ref) for fn, ref in manip_list if ref in test_orig_fns]

    # Split imgs_base and imgs_neg the same way (by index)
    imgs_base_fns = sorted(imgs_base_store.keys())
    rng.shuffle(imgs_base_fns)
    n_train_base = int(len(imgs_base_fns) * TRAIN_RATIO)
    train_imgs_base = {fn: imgs_base_store[fn] for fn in imgs_base_fns[:n_train_base]}
    test_imgs_base  = {fn: imgs_base_store[fn] for fn in imgs_base_fns[n_train_base:]}

    imgs_neg_fns = sorted(imgs_neg_store.keys())
    rng.shuffle(imgs_neg_fns)
    n_train_neg = int(len(imgs_neg_fns) * TRAIN_RATIO)
    train_imgs_neg = [imgs_neg_fns[i] for i in range(n_train_neg)]
    test_imgs_neg  = [imgs_neg_fns[i] for i in range(n_train_neg, len(imgs_neg_fns))]

    # Build base stores (CP originals for that split + imgs base for that split)
    train_base_store = {fn: orig_hashes[fn] for fn in train_orig_fns}
    train_base_store.update(train_imgs_base)
    test_base_store  = {fn: orig_hashes[fn] for fn in test_orig_fns}
    test_base_store.update(test_imgs_base)

    # Build query stores for positives (only hashes, not ref_image)
    train_pos_store = {fn: {h: manip_hashes[fn][h] for h in HASH_NAMES}
                       for fn, _ in train_manips}
    test_pos_store  = {fn: {h: manip_hashes[fn][h] for h in HASH_NAMES}
                       for fn, _ in test_manips}

    # Negative stores
    train_neg_store = {fn: imgs_neg_store[fn] for fn in train_imgs_neg}
    test_neg_store  = {fn: imgs_neg_store[fn] for fn in test_imgs_neg}

    print(f"\n  TRAIN: {len(train_base_store)} base "
          f"({len(train_orig_fns)} CP + {len(train_imgs_base)} imgs)"
          f"  |  {len(train_manips)} positives  |  {len(train_imgs_neg)} negatives")
    print(f"  TEST : {len(test_base_store)} base "
          f"({len(test_orig_fns)} CP + {len(test_imgs_base)} imgs)"
          f"  |  {len(test_manips)} positives  |  {len(test_imgs_neg)} negatives")

    # ---- Compute min-distances ----
    print("\n[3] Computing pairwise min-distances ...")

    print("  Train positives -> train base ...")
    t0 = time.time()
    train_pos_mins = compute_min_distances(
        [fn for fn, _ in train_manips], train_pos_store, train_base_store)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  Train negatives -> train base ...")
    t0 = time.time()
    train_neg_mins = compute_min_distances(
        train_imgs_neg, train_neg_store, train_base_store)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  Test positives -> test base ...")
    t0 = time.time()
    test_pos_mins = compute_min_distances(
        [fn for fn, _ in test_manips], test_pos_store, test_base_store)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  Test negatives -> test base ...")
    t0 = time.time()
    test_neg_mins = compute_min_distances(
        test_imgs_neg, test_neg_store, test_base_store)
    print(f"  Done in {time.time()-t0:.1f}s")

    train_pos_dists = list(train_pos_mins.values())
    train_neg_dists = list(train_neg_mins.values())
    test_pos_dists  = list(test_pos_mins.values())
    test_neg_dists  = list(test_neg_mins.values())

    # ---- Grid search ----
    print("\n" + "=" * 70)
    print("TRAINING PHASE — Brute-force threshold sweep")
    print(f"Objective: Maximize Recall  subject to  FPR <= {MAX_FPR}")
    print("=" * 70)

    best_results: dict[int, tuple[dict, dict]] = {}
    for k in K_VALUES:
        csv_path = RESULTS_DIR / f"train_results_cp_k{k}.csv"
        bt, bm = sweep_thresholds(train_pos_dists, train_neg_dists, k, csv_path)
        best_results[k] = (bt, bm)

    # ---- Training summary ----
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'k':>3}  {'T_aHash':>8}  {'T_pHash':>8}  {'T_hsvHash':>10}  "
          f"{'T_sHash':>8}  {'Recall':>8}  {'FPR':>8}")
    for k in K_VALUES:
        bt, bm = best_results[k]
        if not bt:
            print(f"  k={k}  — no valid config"); continue
        print(f"  k={k}"
              f"  {bt.get('aHash',0):>8}  {bt.get('pHash',0):>8}"
              f"  {bt.get('hsvHash',0):>10}  {bt.get('sHash',0):>8}"
              f"  {bm['Recall']:>8.4f}  {bm['FPR']:>8.4f}")

    # ---- Test phase ----
    print("\n" + "=" * 70)
    print("TEST PHASE  (image-level evaluation)")
    print("=" * 70)
    print(f"  Positive test images: {len(test_pos_dists)}")
    print(f"  Negative test images: {len(test_neg_dists)}")

    report_lines = [
        "CryptoPunk — Minimal Distance Detector Test Report\n",
        "=" * 60 + "\n",
        f"Base tree  : {len(train_base_store)} train / {len(test_base_store)} test\n",
        f"             (CP originals + imgs_base)\n",
        f"Positives  : {len(train_pos_dists)} train / {len(test_pos_dists)} test\n",
        f"Negatives  : {len(train_neg_dists)} train / {len(test_neg_dists)} test\n",
        "=" * 60 + "\n\n",
    ]

    print(f"\n{'k':>3}  {'Recall':>8}  {'FPR':>8}  {'Precision':>10}  "
          f"{'F1':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")

    for k in K_VALUES:
        bt, bm_train = best_results[k]
        if not bt:
            print(f"  k={k}: no valid config"); continue
        m = evaluate_test(test_pos_dists, test_neg_dists, bt, k)
        print(f"  k={k}"
              f"  {m['Recall']:>8.4f}  {m['FPR']:>8.4f}"
              f"  {m['Precision']:>10.4f}  {m['F1']:>8.4f}"
              f"  {m['TP']:>5}  {m['FP']:>5}  {m['FN']:>5}  {m['TN']:>5}")
        report_lines += [
            f"Detector k={k}\n",
            f"  Best thresholds : aHash={bt['aHash']}  pHash={bt['pHash']}"
            f"  hsvHash={bt['hsvHash']}  sHash={bt['sHash']}\n",
            f"  Train Recall={bm_train['Recall']:.4f}  Train FPR={bm_train['FPR']:.4f}\n",
            f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}\n",
            f"  Recall    = {m['Recall']:.4f}\n",
            f"  FPR       = {m['FPR']:.4f}\n",
            f"  Precision = {m['Precision']:.4f}\n",
            f"  F1        = {m['F1']:.4f}\n\n",
        ]

    report_path = RESULTS_DIR / "test_report_cp.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n[PASS]  Report saved to {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
