"""
select_images.py
================
Steps 0 – 3 of the XGBoost training plan:

  Step 0: Select 800 base originals (must have ≥ 1 fake in dataset_duplications)
  Step 1: Take all 700 remaining originals (negatives)
  Step 2: Select 700 fakes whose reference ∈ the 800 base images
  Step 3: 80/20 split of ALL 2 200 images, keeping fake→reference links intact

Outputs CSV manifests into data_manifests/.

Usage
-----
    python src/data_generation/select_images.py
"""

import sys
import random
from pathlib import Path

import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.train_config import (
    BASE_DIR,
    COLLECTIONS_DIRS,
    DUPLICATIONS_METADATA_CSV,
    ORIGINALS_METADATA_CSV,
    DATA_MANIFESTS_DIR,
    N_BASE_IMAGES,
    N_FAKES,
    TRAIN_RATIO,
    RANDOM_SEED,
)


def _resolve_original_path(filename: str, collection: str) -> str:
    """Return the absolute path to an original image given its filename and collection."""
    return str(COLLECTIONS_DIRS[collection] / filename)


def _resolve_fake_path(filename: str) -> str:
    """Return the absolute path to a fake image (search across the 4 sub-folders)."""
    dup_dir = BASE_DIR / "dataset_duplications"
    for sub in ["1_geometric", "2_color", "3_noise_compression", "4_content_modifications"]:
        candidate = dup_dir / sub / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Fake image not found in dataset_duplications/: {filename}")


def run_selection():
    """Execute Steps 0 – 3 and write the six CSV manifests."""
    random.seed(RANDOM_SEED)

    # Create output directory
    DATA_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    originals_df = pd.read_csv(ORIGINALS_METADATA_CSV)
    fakes_df = pd.read_csv(DUPLICATIONS_METADATA_CSV)

    print(f"Loaded {len(originals_df)} originals, {len(fakes_df)} fakes.")

    # Build set of original filenames that have at least one fake
    refs_with_fakes = set(fakes_df["ref_image"].unique())
    print(f"Originals with >= 1 fake: {len(refs_with_fakes)}")

    # ==================================================================
    # STEP 0 - Select 800 base images (must have a fake)
    # ==================================================================
    eligible_base = originals_df[originals_df["filename"].isin(refs_with_fakes)].copy()
    
    base_df = eligible_base.sample(n=N_BASE_IMAGES, random_state=RANDOM_SEED)
    base_filenames = set(base_df["filename"])

    # Add full_path column
    base_df = base_df.copy()
    base_df["full_path"] = base_df.apply(
        lambda r: _resolve_original_path(r["filename"], r["collection"]), axis=1
    )

    base_path = DATA_MANIFESTS_DIR / "base_800.csv"
    base_df.to_csv(base_path, index=False)
    print(f"\nStep 0: Selected {len(base_df)} base images -> {base_path}")
    print(f"  Per collection: {base_df['collection'].value_counts().to_dict()}")

    # ==================================================================
    # STEP 1 - Take all remaining originals (negatives)
    # ==================================================================
    remaining_df = originals_df[~originals_df["filename"].isin(base_filenames)].copy()
    remaining_df["full_path"] = remaining_df.apply(
        lambda r: _resolve_original_path(r["filename"], r["collection"]), axis=1
    )

    orig_path = DATA_MANIFESTS_DIR / "originals_700.csv"
    remaining_df.to_csv(orig_path, index=False)
    print(f"\nStep 1: {len(remaining_df)} remaining originals (negatives) -> {orig_path}")
    print(f"  Per collection: {remaining_df['collection'].value_counts().to_dict()}")

    # ==================================================================
    # STEP 2 - Select 700 fakes whose reference in base set
    # ==================================================================
    eligible_fakes = fakes_df[fakes_df["ref_image"].isin(base_filenames)].copy()
    print(f"\nEligible fakes (ref in base): {len(eligible_fakes)}")

    selected_fakes = eligible_fakes.sample(n=N_FAKES, random_state=RANDOM_SEED)
    selected_fakes = selected_fakes.copy()
    selected_fakes["full_path"] = selected_fakes["filename"].apply(_resolve_fake_path)

    fakes_path = DATA_MANIFESTS_DIR / "fakes_700.csv"
    selected_fakes.to_csv(fakes_path, index=False)
    print(f"Step 2: Selected {len(selected_fakes)} fakes -> {fakes_path}")
    print(f"  Per collection: {selected_fakes['collection'].value_counts().to_dict()}")
    print(f"  Per type: {selected_fakes['type'].value_counts().to_dict()}")

    # ==================================================================
    # STEP 3 - 80/20 split of ALL 2 200 images
    # ==================================================================
    #
    # Split order:
    #   1. Split the 800 BASE images 80/20 (stratified by collection)
    #   2. Assign fakes to whichever split their reference landed in
    #   3. Split the 700 ORIGINALS 80/20 (stratified by collection)
    #
    # This guarantees that every fake's reference is in its own split's
    # base tree - no data leakage.
    # ------------------------------------------------------------------

    n_train_base = int(N_BASE_IMAGES * TRAIN_RATIO)  # 640

    # 3a. Split base images (stratified by collection)
    train_base = base_df.groupby("collection", group_keys=False).apply(
        lambda g: g.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
    )
    test_base = base_df.drop(train_base.index)

    train_base_filenames = set(train_base["filename"])
    test_base_filenames = set(test_base["filename"])

    # 3b. Assign fakes by reference
    train_fakes = selected_fakes[selected_fakes["ref_image"].isin(train_base_filenames)].copy()
    test_fakes  = selected_fakes[selected_fakes["ref_image"].isin(test_base_filenames)].copy()

    # 3c. Split originals (stratified by collection)
    train_originals = remaining_df.groupby("collection", group_keys=False).apply(
        lambda g: g.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
    )
    test_originals = remaining_df.drop(train_originals.index)

    # ------------------------------------------------------------------
    # Save all 6 CSVs
    # ------------------------------------------------------------------
    manifests = {
        "train_base.csv":      train_base,
        "test_base.csv":       test_base,
        "train_originals.csv": train_originals,
        "test_originals.csv":  test_originals,
        "train_fakes.csv":     train_fakes,
        "test_fakes.csv":      test_fakes,
    }

    print(f"\n{'='*60}")
    print("Step 3: 80/20 split results")
    print(f"{'='*60}")

    total_train = len(train_base) + len(train_originals) + len(train_fakes)
    total_test  = len(test_base)  + len(test_originals)  + len(test_fakes)

    print(f"\n{'Component':<22} {'Train':>8} {'Test':>8} {'Total':>8}")
    print(f"{'-'*48}")
    print(f"{'Base images':<22} {len(train_base):>8} {len(test_base):>8} {len(base_df):>8}")
    print(f"{'Originals (negatives)':<22} {len(train_originals):>8} {len(test_originals):>8} {len(remaining_df):>8}")
    print(f"{'Fakes (positives)':<22} {len(train_fakes):>8} {len(test_fakes):>8} {len(selected_fakes):>8}")
    print(f"{'-'*48}")
    print(f"{'TOTAL':<22} {total_train:>8} {total_test:>8} {total_train + total_test:>8}")

    for name, df in manifests.items():
        path = DATA_MANIFESTS_DIR / name
        df.to_csv(path, index=False)
        print(f"  -> {path}  ({len(df)} rows)")

    print(f"\n[OK] All manifests saved to {DATA_MANIFESTS_DIR}/")
    return manifests


if __name__ == "__main__":
    run_selection()
