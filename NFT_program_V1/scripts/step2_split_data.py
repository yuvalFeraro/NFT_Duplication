"""
step2_split_data.py
===================
Splits the DISC-21 dataset into train/test splits, saving 6 CSV manifests:
    train_base.csv          - 80% of the 516 reference images that have a query pair
    train_negatives.csv     - 80% of the 516 reference images with NO query pair
    train_duplications.csv  - 80% of the 516 query images (coupled with train_base)
    test_base.csv           - remaining 20% of paired references
    test_negatives.csv      - remaining 20% of negatives
    test_duplications.csv   - remaining 20% of queries (coupled with test_base)

Split rule: a query and its reference ALWAYS land in the same split.
Negatives are split independently (no coupling constraint).

Run:
    python step2_split_data.py
"""

import random
import shutil
import tempfile
from pathlib import Path

import openpyxl

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_V2  = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
BASE_DATA = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets")

DISC21_REFS_DIR    = BASE_V2 / "Data_sets" / "DISC-21"
DISC21_QUERIES_DIR = BASE_V2 / "dataset_duplications" / "DISC21_duplications"
MAPPING_XLSX       = BASE_DATA / "DISC-21" / "2409_data" / "combined_queries_distances.xlsx"
OUT_DIR            = Path(__file__).resolve().parents[1] / "data" / "splits"

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_xlsx(xlsx_path: Path) -> list:
    """Copy xlsx to a temp file (avoids Windows file-lock), parse it, return rows."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(xlsx_path, tmp_path)
        wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        return rows
    finally:
        tmp_path.unlink(missing_ok=True)


def _save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """Write a list of dicts to a CSV file."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Saved {len(rows):4d} rows → {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split():
    print("=" * 60)
    print("STEP 2 - Dataset Splitting (DISC-21)")
    print("=" * 60)

    # 1. Load mapping
    rows = _safe_load_xlsx(MAPPING_XLSX)
    header = rows[0]
    data_rows = rows[1:]
    qi = header.index("query_filename")
    ri = header.index("reference_filename")
    # Each element: (query_filename, reference_filename)
    pairs: list[tuple[str, str]] = [(r[qi], r[ri]) for r in data_rows]
    print(f"\n[1] Loaded {len(pairs)} query-reference pairs.")

    # 2. Identify negatives (refs on disk that have NO query pair)
    ref_files_on_disk = set(
        f.name for f in DISC21_REFS_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    paired_refs = set(p[1] for p in pairs)
    negatives: list[str] = sorted(ref_files_on_disk - paired_refs)
    print(f"[2] Found {len(negatives)} negative reference images (no query pair).")

    # 3. Shuffle with fixed seed for reproducibility
    rng = random.Random(RANDOM_SEED)
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)
    shuffled_negs = list(negatives)
    rng.shuffle(shuffled_negs)

    # 4. Split
    n_pairs = len(shuffled_pairs)
    n_train_pairs = round(n_pairs * TRAIN_RATIO)  # 413

    train_pairs = shuffled_pairs[:n_train_pairs]
    test_pairs  = shuffled_pairs[n_train_pairs:]

    n_negs = len(shuffled_negs)
    n_train_negs = round(n_negs * TRAIN_RATIO)  # 413

    train_negs = shuffled_negs[:n_train_negs]
    test_negs  = shuffled_negs[n_train_negs:]

    print(f"\n[3] Split summary:")
    print(f"    Train pairs (base+duplications) : {len(train_pairs)}")
    print(f"    Test  pairs (base+duplications) : {len(test_pairs)}")
    print(f"    Train negatives                 : {len(train_negs)}")
    print(f"    Test  negatives                 : {len(test_negs)}")

    # 5. Save CSVs
    print(f"\n[4] Saving split CSVs to: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # train_base
    _save_csv(
        [{"filename": ref, "query_filename": q} for q, ref in train_pairs],
        OUT_DIR / "train_base.csv",
        ["filename", "query_filename"]
    )
    # test_base
    _save_csv(
        [{"filename": ref, "query_filename": q} for q, ref in test_pairs],
        OUT_DIR / "test_base.csv",
        ["filename", "query_filename"]
    )
    # train_duplications
    _save_csv(
        [{"filename": q, "reference_filename": ref} for q, ref in train_pairs],
        OUT_DIR / "train_duplications.csv",
        ["filename", "reference_filename"]
    )
    # test_duplications
    _save_csv(
        [{"filename": q, "reference_filename": ref} for q, ref in test_pairs],
        OUT_DIR / "test_duplications.csv",
        ["filename", "reference_filename"]
    )
    # train_negatives
    _save_csv(
        [{"filename": f} for f in train_negs],
        OUT_DIR / "train_negatives.csv",
        ["filename"]
    )
    # test_negatives
    _save_csv(
        [{"filename": f} for f in test_negs],
        OUT_DIR / "test_negatives.csv",
        ["filename"]
    )

    print("\n[PASS]  Splitting complete.")
    print("=" * 60)


if __name__ == "__main__":
    split()
