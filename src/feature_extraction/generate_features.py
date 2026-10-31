"""
generate_features.py
====================
Steps 4 - 5 of the XGBoost training plan:

  Step 4: Build two separate BK-Trees (train-tree from train-base, test-tree from test-base)
          using PRE-COMPUTED hashes from the metadata CSVs.
  Step 5: For each query image (originals + fakes), retrieve 50-NN candidates from
          the appropriate BK-Tree, compute distance vectors + optional rank features,
          and assign labels.

Usage
-----
    python src/feature_extraction/generate_features.py                # V1 (distances only)
    python src/feature_extraction/generate_features.py --version v2   # V2 (distances + ranks)
"""

import sys
import time
import argparse
from pathlib import Path

import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.train_config import (
    DATA_MANIFESTS_DIR,
    BKTREES_DIR,
    ORIGINALS_METADATA_CSV,
    DUPLICATIONS_METADATA_CSV,
    K_NEAREST,
    BK_TREE_DISTANCE_THRESHOLD,
    LABEL_COLUMN,
    get_version_config,
)
from src.bktree.bk_tree import (
    build_trees_from_hashes,
    save_trees,
    compute_distance_vector,
    parse_hash_from_hex,
)

# Hash columns in the metadata CSVs
HASH_COLUMNS = ["aHash", "pHash", "sHash", "hsvHash"]
NOT_FOUND_RANK = K_NEAREST + 1  # 51 — used when candidate not in a tree's top-K


def _load_manifest(name: str) -> pd.DataFrame:
    """Load a CSV manifest from data_manifests/."""
    path = DATA_MANIFESTS_DIR / name
    df = pd.read_csv(path)
    return df


def _build_hash_lookup() -> dict:
    """
    Build a unified lookup: filename -> {aHash: hex, pHash: hex, ...}
    from the originals + duplications metadata CSVs.
    """
    orig_df = pd.read_csv(ORIGINALS_METADATA_CSV)
    fakes_df = pd.read_csv(DUPLICATIONS_METADATA_CSV)

    lookup = {}
    for _, row in orig_df.iterrows():
        lookup[row["filename"]] = {h: str(row[h]) for h in HASH_COLUMNS}
    for _, row in fakes_df.iterrows():
        lookup[row["filename"]] = {h: str(row[h]) for h in HASH_COLUMNS}

    print(f"  Hash lookup built: {len(lookup)} images")
    return lookup


def _query_trees_with_ranks(trees, q_hashes, k):
    """
    Query each BK-Tree for top-K, return:
      - candidate_set: dict {candidate_name: {hash_name: (rank, distance), ...}}
    """
    candidate_info = {}

    for hash_name, tree in trees.items():
        results = tree.top_k(q_hashes[hash_name], k=k)
        for rank_idx, (dist, name) in enumerate(results):
            if name not in candidate_info:
                candidate_info[name] = {}
            candidate_info[name][hash_name] = (rank_idx + 1, dist)  # 1-indexed rank

    return candidate_info


def _extract_features_for_split(
    split_name: str,
    base_csv: str,
    originals_csv: str,
    fakes_csv: str,
    tree_prefix: str,
    hash_lookup: dict,
    include_ranks: bool = False,
) -> pd.DataFrame:
    """
    Build a BK-Tree from base images (using pre-computed hashes),
    then extract features for originals + fakes.

    Parameters
    ----------
    include_ranks : bool
        If True, include rank_aHash, rank_pHash, rank_sHash, rank_hsvHash,
        n_trees_found, best_rank columns (V2 features).
    """
    print(f"\n{'='*60}")
    print(f" {split_name.upper()} SPLIT - Feature Extraction")
    print(f"{'='*60}")

    # Load manifests
    base_df = _load_manifest(base_csv)
    originals_df = _load_manifest(originals_csv)
    fakes_df = _load_manifest(fakes_csv)

    # ------------------------------------------------------------------
    # Step 4: Build BK-Tree from base images (pre-computed hashes)
    # ------------------------------------------------------------------
    print(f"\nStep 4: Building {split_name} BK-Trees from {len(base_df)} base images ...")
    start = time.time()

    base_rows = base_df[["filename"] + HASH_COLUMNS].to_dict("records")

    trees, hash_store = build_trees_from_hashes(
        base_rows,
        distance_threshold=BK_TREE_DISTANCE_THRESHOLD,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"  BK-Trees built in {elapsed:.1f}s")

    # Save trees for later use (pipeline deployment)
    save_trees(trees, hash_store, BKTREES_DIR, prefix=tree_prefix)

    # ------------------------------------------------------------------
    # Step 5: Extract features for originals + fakes
    # ------------------------------------------------------------------
    version_label = "V2 (distances + ranks)" if include_ranks else "V1 (distances only)"
    print(f"\nStep 5: Extracting {version_label} features for {split_name} queries ...")

    all_feature_rows = []

    # Combine query images: originals (label always 0) + fakes (label 1 for the true ref)
    query_list = []

    for _, row in originals_df.iterrows():
        query_list.append({
            "filename": row["filename"],
            "is_fake": False,
            "ref_image": None,
        })

    for _, row in fakes_df.iterrows():
        query_list.append({
            "filename": row["filename"],
            "is_fake": True,
            "ref_image": row["ref_image"],
        })

    total_queries = len(query_list)
    ref_found_count = 0
    ref_missed_count = 0

    for q_idx, query in enumerate(query_list):
        fn = query["filename"]

        # Get pre-computed hashes for the query image
        if fn not in hash_lookup:
            print(f"  WARNING: {fn} not in hash lookup, skipping")
            continue

        hex_hashes = hash_lookup[fn]

        # Convert hex strings to imagehash objects
        q_hashes = {}
        for hash_name in HASH_COLUMNS:
            q_hashes[hash_name] = parse_hash_from_hex(hash_name, hex_hashes[hash_name])

        # Query BK-Trees with rank tracking
        candidate_info = _query_trees_with_ranks(trees, q_hashes, k=K_NEAREST)

        # Track whether the true reference was found in candidates
        ref_in_candidates = False

        for candidate_name, tree_hits in candidate_info.items():
            # Skip self-match
            if candidate_name == query["filename"]:
                continue

            # Compute distance vector
            if candidate_name not in hash_store:
                continue

            dist_vec = compute_distance_vector(q_hashes, hash_store[candidate_name])

            # Assign label
            if query["is_fake"] and candidate_name == query["ref_image"]:
                label = 1
                ref_in_candidates = True
            else:
                label = 0

            row = {
                "query_image": query["filename"],
                "candidate_image": candidate_name,
                **dist_vec,
                LABEL_COLUMN: label,
            }

            # Add rank features if V2
            if include_ranks:
                for hash_name in HASH_COLUMNS:
                    if hash_name in tree_hits:
                        row[f"rank_{hash_name}"] = tree_hits[hash_name][0]  # rank
                    else:
                        row[f"rank_{hash_name}"] = NOT_FOUND_RANK  # 51

                row["n_trees_found"] = sum(1 for h in HASH_COLUMNS if h in tree_hits)
                ranks = [tree_hits[h][0] for h in HASH_COLUMNS if h in tree_hits]
                row["best_rank"] = min(ranks) if ranks else NOT_FOUND_RANK

            all_feature_rows.append(row)

        # Stats tracking
        if query["is_fake"]:
            if ref_in_candidates:
                ref_found_count += 1
            else:
                ref_missed_count += 1

        if (q_idx + 1) % 100 == 0 or q_idx + 1 == total_queries:
            print(f"  Processed {q_idx + 1}/{total_queries} queries ...")

    # Build DataFrame
    features_df = pd.DataFrame(all_feature_rows)

    # Summary
    n_positive = (features_df[LABEL_COLUMN] == 1).sum() if len(features_df) > 0 else 0
    n_negative = (features_df[LABEL_COLUMN] == 0).sum() if len(features_df) > 0 else 0
    total_fakes = len(fakes_df)

    print(f"\n  {split_name} feature extraction summary:")
    print(f"    Total feature rows:  {len(features_df)}")
    print(f"    Positive (label=1):  {n_positive}")
    print(f"    Negative (label=0):  {n_negative}")
    print(f"    Imbalance ratio:     1:{n_negative // max(n_positive, 1)}")
    print(f"    Reference found in 50-NN: {ref_found_count}/{total_fakes} "
          f"({100 * ref_found_count / max(total_fakes, 1):.1f}%)")
    print(f"    Reference missed:         {ref_missed_count}/{total_fakes} "
          f"({100 * ref_missed_count / max(total_fakes, 1):.1f}%)")

    if include_ranks and n_positive > 0:
        pos_rows = features_df[features_df[LABEL_COLUMN] == 1]
        print(f"\n    Positive pairs rank summary:")
        for h in HASH_COLUMNS:
            med = pos_rows[f"rank_{h}"].median()
            pct_rank1 = (pos_rows[f"rank_{h}"] == 1).mean() * 100
            print(f"      rank_{h}: median={med:.0f}, rank=1 in {pct_rank1:.1f}%")
        print(f"      n_trees_found: median={pos_rows['n_trees_found'].median():.0f}")
        print(f"      best_rank: median={pos_rows['best_rank'].median():.0f}")

    return features_df


def run_feature_extraction(version: str = "v1"):
    """Run feature extraction for both train and test splits."""
    vcfg = get_version_config(version)
    features_dir = vcfg["features_dir"]
    include_ranks = (version == "v2")

    features_dir.mkdir(parents=True, exist_ok=True)
    BKTREES_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Build unified hash lookup from pre-computed metadata
    print(f"[{version.upper()}] Loading pre-computed hashes from metadata CSVs ...")
    hash_lookup = _build_hash_lookup()

    # -- Train split --
    train_features = _extract_features_for_split(
        split_name="train",
        base_csv="train_base.csv",
        originals_csv="train_originals.csv",
        fakes_csv="train_fakes.csv",
        tree_prefix="train",
        hash_lookup=hash_lookup,
        include_ranks=include_ranks,
    )

    train_path = features_dir / "train_features.csv"
    train_features.to_csv(train_path, index=False)
    print(f"\n  -> Saved {train_path}")

    # -- Test split --
    test_features = _extract_features_for_split(
        split_name="test",
        base_csv="test_base.csv",
        originals_csv="test_originals.csv",
        fakes_csv="test_fakes.csv",
        tree_prefix="test",
        hash_lookup=hash_lookup,
        include_ranks=include_ranks,
    )

    test_path = features_dir / "test_features.csv"
    test_features.to_csv(test_path, index=False)
    print(f"\n  -> Saved {test_path}")

    # -- Save Excel metadata for V2 --
    if include_ranks:
        xlsx_path = features_dir / "features_v2_metadata.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            train_features.to_excel(writer, sheet_name="train_features", index=False)
            test_features.to_excel(writer, sheet_name="test_features", index=False)

            # Summary sheet
            summary_data = {
                "metric": [
                    "train_total_rows", "train_positives", "train_negatives",
                    "test_total_rows", "test_positives", "test_negatives",
                    "feature_count", "features",
                ],
                "value": [
                    len(train_features),
                    int((train_features[LABEL_COLUMN] == 1).sum()),
                    int((train_features[LABEL_COLUMN] == 0).sum()),
                    len(test_features),
                    int((test_features[LABEL_COLUMN] == 1).sum()),
                    int((test_features[LABEL_COLUMN] == 0).sum()),
                    len(vcfg["feature_columns"]),
                    ", ".join(vcfg["feature_columns"]),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)
        print(f"  -> Saved Excel metadata: {xlsx_path}")

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f" [{version.upper()}] Feature extraction complete in {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features for XGBoost training")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1",
                        help="v1 = distances only (4 features), v2 = distances + ranks (10 features)")
    args = parser.parse_args()
    run_feature_extraction(version=args.version)
