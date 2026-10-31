"""
step5_generate_features.py
==========================
Build BK-Trees from the DISC-21 base sets and extract V2 features
(4 Hamming distances + 4 ranks + 2 summary = 10 features per pair)
for training and test splits.

Logic (mirrors NFT_program_V2/src/feature_extraction/generate_features.py):

  For each split (train / test):
    1. Build 4 BK-Trees from the BASE set (reference images).
    2. For each DUPLICATE query image:
         - Query trees for top-K candidates.
         - For every candidate in the union:
             * (query, true_reference)    -> label = 1
             * (query, any_other_cand)    -> label = 0
    3. For each NEGATIVE image (refs not paired with any query):
         - Query the same trees for top-K candidates.
         - All (negative, candidate) pairs -> label = 0

Feature vector per row (V2, 10 features):
    dist_aHash, dist_pHash, dist_hsvHash, dist_sHash   (Hamming distances)
    rank_aHash, rank_pHash, rank_hsvHash, rank_sHash   (1-indexed top-K rank)
    n_trees_found                                       (# trees where candidate appeared)
    best_rank                                           (min rank across trees)

Outputs (in NFT_program_V1/data/features/):
    train_features.csv
    test_features.csv

Run:
    python step5_generate_features.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
V1_DIR       = SCRIPT_DIR.parent
SPLITS_DIR   = V1_DIR / "data" / "splits"
HASHES_DIR   = V1_DIR / "data" / "hashes"
FEATURES_DIR = V1_DIR / "data" / "features"

# BK-Tree implementation lives in the NFT_program_V2 src folder
V2_DIR = V1_DIR.parent
sys.path.insert(0, str(V2_DIR))

# Sibling project (needed by bk_tree.py to import BKTree & distance functions)
_SIBLING_HASH_DIR = (
    V2_DIR.parent
    / "NFT_program"
    / "NFT-Duplications-detection-using-Image-hash-functions-main"
)
sys.path.insert(0, str(_SIBLING_HASH_DIR))

from src.bktree.bk_tree import (       # noqa: E402
    build_trees_from_hashes,
    compute_distance_vector,
    parse_hash_from_hex,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HASH_COLUMNS  = ["aHash", "pHash", "hsvHash", "sHash"]
K_NEAREST     = 50
NOT_FOUND_RANK = K_NEAREST + 1   # 51 — used when candidate not in a tree's top-K
LABEL_COLUMN  = "label"
BK_TREE_DIST_THRESHOLD = 100     # large; top_k ignores it


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_hashes(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Return {filename: {hash_name: parsed_hash_object}}."""
    records = load_csv(csv_path)
    store: dict[str, dict[str, Any]] = {}
    for row in records:
        fn = row["filename"].strip()
        store[fn] = {
            h: parse_hash_from_hex(h, row[h])
            for h in HASH_COLUMNS
        }
    return store


def query_trees_with_ranks(
    trees: dict,
    q_hashes: dict[str, Any],
    k: int,
) -> dict[str, dict[str, tuple[int, float]]]:
    """
    Query each BK-Tree for top-K, return per-candidate rank+distance info.

    Returns
    -------
    candidate_info : {candidate_name: {hash_name: (rank_1indexed, distance)}}
    """
    candidate_info: dict[str, dict[str, tuple[int, float]]] = {}
    for hash_name, tree in trees.items():
        results = tree.top_k(q_hashes[hash_name], k=k)
        for rank_idx, (dist, name) in enumerate(results):
            if name not in candidate_info:
                candidate_info[name] = {}
            candidate_info[name][hash_name] = (rank_idx + 1, dist)
    return candidate_info


# ---------------------------------------------------------------------------
# Per-split feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_split(
    split_name: str,
    base_hash_rows: list[dict],
    dup_rows: list[dict],        # [{filename, reference_filename}, ...]
    neg_fns: list[str],
    query_hash_store: dict,      # pre-loaded hashes for duplicate queries
    neg_hash_store: dict,        # pre-loaded hashes for negatives
) -> pd.DataFrame:
    """
    Build a BK-Tree from `base_hash_rows`, then extract V2 features for
    duplicate queries and negatives.
    """
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SPLIT — Feature Extraction")
    print(f"{'='*60}")

    # ---- Build BK-Trees from base set ----
    print(f"\nBuilding {split_name} BK-Trees from {len(base_hash_rows)} base images ...")
    t0 = time.time()
    trees, hash_store = build_trees_from_hashes(
        base_hash_rows,
        distance_threshold=BK_TREE_DIST_THRESHOLD,
        verbose=True,
    )
    print(f"  BK-Trees built in {time.time()-t0:.1f}s")

    all_rows: list[dict] = []
    ref_found   = 0
    ref_missed  = 0

    # ---- Duplicate queries ----
    print(f"\nExtracting features for {len(dup_rows)} duplicate queries ...")
    for i, dup in enumerate(dup_rows):
        q_fn  = dup["filename"].strip()
        ref_fn = dup["reference_filename"].strip()

        if q_fn not in query_hash_store:
            print(f"  WARNING: query {q_fn} not in hash store, skipping")
            continue

        q_hashes = query_hash_store[q_fn]
        candidate_info = query_trees_with_ranks(trees, q_hashes, k=K_NEAREST)

        found_ref = False
        for cand_fn, tree_hits in candidate_info.items():
            if cand_fn == q_fn:
                continue  # skip self-match
            if cand_fn not in hash_store:
                continue

            dist_vec = compute_distance_vector(q_hashes, hash_store[cand_fn])
            label = 1 if cand_fn == ref_fn else 0
            if label == 1:
                found_ref = True

            # Rank features
            ranks = []
            row = {
                "query_image":     q_fn,
                "candidate_image": cand_fn,
                **dist_vec,
                LABEL_COLUMN:      label,
            }
            for h in HASH_COLUMNS:
                if h in tree_hits:
                    row[f"rank_{h}"] = tree_hits[h][0]
                    ranks.append(tree_hits[h][0])
                else:
                    row[f"rank_{h}"] = NOT_FOUND_RANK

            row["n_trees_found"] = sum(1 for h in HASH_COLUMNS if h in tree_hits)
            row["best_rank"]     = min(ranks) if ranks else NOT_FOUND_RANK

            all_rows.append(row)

        if found_ref:
            ref_found += 1
        else:
            ref_missed += 1

        if (i + 1) % 50 == 0 or i + 1 == len(dup_rows):
            print(f"  Processed {i+1}/{len(dup_rows)} duplicate queries ...")

    # ---- Negatives ----
    print(f"\nExtracting features for {len(neg_fns)} negatives ...")
    for i, neg_fn in enumerate(neg_fns):
        if neg_fn not in neg_hash_store:
            print(f"  WARNING: negative {neg_fn} not in hash store, skipping")
            continue

        q_hashes = neg_hash_store[neg_fn]
        candidate_info = query_trees_with_ranks(trees, q_hashes, k=K_NEAREST)

        for cand_fn, tree_hits in candidate_info.items():
            if cand_fn == neg_fn:
                continue
            if cand_fn not in hash_store:
                continue

            dist_vec = compute_distance_vector(q_hashes, hash_store[cand_fn])

            ranks = []
            row = {
                "query_image":     neg_fn,
                "candidate_image": cand_fn,
                **dist_vec,
                LABEL_COLUMN:      0,
            }
            for h in HASH_COLUMNS:
                if h in tree_hits:
                    row[f"rank_{h}"] = tree_hits[h][0]
                    ranks.append(tree_hits[h][0])
                else:
                    row[f"rank_{h}"] = NOT_FOUND_RANK

            row["n_trees_found"] = sum(1 for h in HASH_COLUMNS if h in tree_hits)
            row["best_rank"]     = min(ranks) if ranks else NOT_FOUND_RANK

            all_rows.append(row)

        if (i + 1) % 50 == 0 or i + 1 == len(neg_fns):
            print(f"  Processed {i+1}/{len(neg_fns)} negatives ...")

    # ---- Summary ----
    df = pd.DataFrame(all_rows)
    n_pos = int((df[LABEL_COLUMN] == 1).sum()) if len(df) > 0 else 0
    n_neg = int((df[LABEL_COLUMN] == 0).sum()) if len(df) > 0 else 0

    print(f"\n  {split_name} extraction summary:")
    print(f"    Total feature rows : {len(df)}")
    print(f"    Positives (label=1): {n_pos}")
    print(f"    Negatives (label=0): {n_neg}")
    print(f"    Imbalance ratio    : 1:{n_neg // max(n_pos, 1)}")
    print(f"    Reference found in top-{K_NEAREST}: "
          f"{ref_found}/{len(dup_rows)} ({100*ref_found/max(len(dup_rows),1):.1f}%)")
    print(f"    Reference missed              : "
          f"{ref_missed}/{len(dup_rows)} ({100*ref_missed/max(len(dup_rows),1):.1f}%)")

    if n_pos > 0:
        pos_rows = df[df[LABEL_COLUMN] == 1]
        print(f"\n    Positive-pair rank summary:")
        for h in HASH_COLUMNS:
            col = f"rank_{h}"
            med = pos_rows[col].median()
            pct1 = (pos_rows[col] == 1).mean() * 100
            print(f"      {col}: median={med:.0f},  rank=1 in {pct1:.1f}%")
        print(f"      best_rank:    median={pos_rows['best_rank'].median():.0f}")
        print(f"      n_trees_found: median={pos_rows['n_trees_found'].median():.0f}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 5 — BK-Tree Feature Extraction (V2: distances + ranks)")
    print("=" * 70)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load split manifests ----
    print("\n[1] Loading split manifests ...")
    train_base_rows = load_csv(SPLITS_DIR / "train_base.csv")
    test_base_rows  = load_csv(SPLITS_DIR / "test_base.csv")
    train_dup_rows  = load_csv(SPLITS_DIR / "train_duplications.csv")
    test_dup_rows   = load_csv(SPLITS_DIR / "test_duplications.csv")
    train_neg_fns   = [r["filename"] for r in load_csv(SPLITS_DIR / "train_negatives.csv")]
    test_neg_fns    = [r["filename"] for r in load_csv(SPLITS_DIR / "test_negatives.csv")]

    print(f"  Train: {len(train_base_rows)} base | "
          f"{len(train_dup_rows)} duplications | {len(train_neg_fns)} negatives")
    print(f"  Test : {len(test_base_rows)} base  | "
          f"{len(test_dup_rows)} duplications  | {len(test_neg_fns)} negatives")

    # ---- Load pre-computed hashes ----
    print("\n[2] Loading pre-computed hashes ...")
    ref_store = load_hashes(HASHES_DIR / "references_hashes.csv")
    q_store   = load_hashes(HASHES_DIR / "queries_hashes.csv")
    neg_store = load_hashes(HASHES_DIR / "negatives_hashes.csv")
    print(f"  References : {len(ref_store)}")
    print(f"  Queries    : {len(q_store)}")
    print(f"  Negatives  : {len(neg_store)}")

    # Enrich base rows with hash columns from ref_store
    def enrich_base_rows(base_rows: list[dict]) -> list[dict]:
        enriched = []
        for r in base_rows:
            fn = r["filename"].strip()
            if fn not in ref_store:
                print(f"  WARNING: base image {fn} not in references hash store, skipping")
                continue
            row = {"filename": fn}
            for h in HASH_COLUMNS:
                # Need raw hex strings for build_trees_from_hashes
                row[h] = _raw_hex(fn, h)
            enriched.append(row)
        return enriched

    # We need the raw hex strings for BK-Tree construction (it parses internally)
    # Load raw hex from CSV directly
    def load_raw_hex(csv_path: Path) -> dict[str, dict[str, str]]:
        records = load_csv(csv_path)
        return {r["filename"].strip(): {h: r[h] for h in HASH_COLUMNS} for r in records}

    raw_ref_hex = load_raw_hex(HASHES_DIR / "references_hashes.csv")

    def make_base_rows(base_rows: list[dict]) -> list[dict]:
        result = []
        for r in base_rows:
            fn = r["filename"].strip()
            if fn not in raw_ref_hex:
                print(f"  WARNING: {fn} missing from references_hashes.csv")
                continue
            result.append({"filename": fn, **raw_ref_hex[fn]})
        return result

    train_base_image_rows = make_base_rows(train_base_rows)
    test_base_image_rows  = make_base_rows(test_base_rows)

    # ---- Extract features ----
    start = time.time()

    train_df = extract_features_for_split(
        split_name="Train",
        base_hash_rows=train_base_image_rows,
        dup_rows=train_dup_rows,
        neg_fns=train_neg_fns,
        query_hash_store=q_store,
        neg_hash_store=neg_store,
    )

    test_df = extract_features_for_split(
        split_name="Test",
        base_hash_rows=test_base_image_rows,
        dup_rows=test_dup_rows,
        neg_fns=test_neg_fns,
        query_hash_store=q_store,
        neg_hash_store=neg_store,
    )

    # ---- Save ----
    print("\n[3] Saving feature files ...")
    train_path = FEATURES_DIR / "train_features.csv"
    test_path  = FEATURES_DIR / "test_features.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  -> {train_path}  ({len(train_df)} rows)")
    print(f"  -> {test_path}   ({len(test_df)} rows)")

    total = time.time() - start
    print(f"\n{'='*70}")
    print(f"  Feature extraction complete in {total:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
