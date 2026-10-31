"""
step1_verify_data.py
====================
Verifies the integrity of the DISC-21 dataset before any training:
  1. Loads combined_queries_distances.xlsx (query→reference mapping, 516 rows).
  2. Checks every reference in the mapping exists in Data_sets/DISC-21/ (1032 images).
  3. Checks every query file in DISC21_duplications/ appears in the mapping.
  4. Prints a clean summary report.

Run:
    python step1_verify_data.py
"""

import os
import shutil
import tempfile
from pathlib import Path

import openpyxl

# ---------------------------------------------------------------------------
# PATHS  (adjust if project root changes)
# ---------------------------------------------------------------------------
BASE_V2 = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
BASE_DATA = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets")

DISC21_REFS_DIR   = BASE_V2 / "Data_sets" / "DISC-21"
DISC21_QUERIES_DIR = BASE_V2 / "dataset_duplications" / "DISC21_duplications"
MAPPING_XLSX      = BASE_DATA / "DISC-21" / "2409_data" / "combined_queries_distances.xlsx"


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


def verify():
    print("=" * 60)
    print("STEP 1 - Data Verification for DISC-21")
    print("=" * 60)

    # --- 1. Load the mapping ---
    print(f"\n[1] Loading mapping from:\n    {MAPPING_XLSX}")
    rows = _safe_load_xlsx(MAPPING_XLSX)
    header = rows[0]
    data_rows = rows[1:]
    print(f"    Header : {header}")
    print(f"    Rows   : {len(data_rows)}")

    # Build (query→reference) dict from mapping
    qi = header.index("query_filename")
    ri = header.index("reference_filename")
    mapping: dict[str, str] = {r[qi]: r[ri] for r in data_rows}

    # --- 2. Scan DISC-21 reference folder ---
    print(f"\n[2] Scanning reference folder:\n    {DISC21_REFS_DIR}")
    if not DISC21_REFS_DIR.exists():
        print(f"    ERROR: folder does not exist!")
        return
    ref_files_on_disk = set(f.name for f in DISC21_REFS_DIR.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"    Found {len(ref_files_on_disk)} reference images on disk.")

    # --- 3. Scan DISC-21 duplications (query) folder ---
    print(f"\n[3] Scanning query folder:\n    {DISC21_QUERIES_DIR}")
    if not DISC21_QUERIES_DIR.exists():
        print(f"    ERROR: folder does not exist!")
        return
    query_files_on_disk = set(f.name for f in DISC21_QUERIES_DIR.iterdir()
                              if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"    Found {len(query_files_on_disk)} query images on disk.")

    # --- 4. Cross-check ---
    print("\n[4] Cross-checking...")

    # 4a. References in mapping vs. files on disk
    refs_in_mapping = set(mapping.values())
    missing_refs = refs_in_mapping - ref_files_on_disk
    extra_refs_disk = ref_files_on_disk - refs_in_mapping
    refs_with_no_duplication = extra_refs_disk  # these are the 516 negatives

    print(f"\n    References in mapping        : {len(refs_in_mapping)}")
    print(f"    References found on disk     : {len(ref_files_on_disk)}")
    print(f"    References MISSING from disk : {len(missing_refs)}")
    if missing_refs:
        for m in sorted(missing_refs):
            print(f"        MISSING: {m}")

    print(f"\n    References on disk with NO query pair (negatives): {len(refs_with_no_duplication)}")

    # 4b. Queries in mapping vs. files on disk
    queries_in_mapping = set(mapping.keys())
    missing_queries = queries_in_mapping - query_files_on_disk
    extra_queries_disk = query_files_on_disk - queries_in_mapping

    print(f"\n    Queries in mapping           : {len(queries_in_mapping)}")
    print(f"    Queries found on disk        : {len(query_files_on_disk)}")
    print(f"    Queries MISSING from disk    : {len(missing_queries)}")
    if missing_queries:
        for m in sorted(missing_queries):
            print(f"        MISSING: {m}")
    print(f"    Queries on disk NOT in mapping: {len(extra_queries_disk)}")
    if extra_queries_disk:
        for m in sorted(extra_queries_disk):
            print(f"        EXTRA: {m}")

    # --- 5. Summary ---
    print("\n" + "=" * 60)
    ok = (len(missing_refs) == 0 and len(missing_queries) == 0)
    if ok:
        print("[PASS] VERIFICATION PASSED")
        print(f"    516 query-reference pairs confirmed.")
        print(f"    {len(refs_with_no_duplication)} reference images with no pair (will become negatives).")
    else:
        print("[FAIL] VERIFICATION FAILED - see issues above.")
    print("=" * 60)


if __name__ == "__main__":
    verify()
