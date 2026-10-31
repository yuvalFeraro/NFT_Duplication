"""
step3_extract_hashes.py
=======================
Loads hash values for all 516 reference images and 516 query images from the
pre-computed xlsx files in:
    data_sets/DISC-21/2409_data/

Parses and normalises:
    aHash, pHash, hsvHash  → hex string  (stored as-is; distance computed by XOR-popcount)
    sHash                  → comma-separated hex segments (stored as-is)

Saves two clean CSVs:
    data/hashes/references_hashes.csv   - 516 rows
    data/hashes/queries_hashes.csv      - 516 rows

Both CSVs have columns: filename, aHash, pHash, hsvHash, sHash

Run:
    python step3_extract_hashes.py
"""

import csv
import shutil
import tempfile
from pathlib import Path

import openpyxl

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DATA = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets")
DATA_2409  = BASE_DATA / "DISC-21" / "2409_data"
OUT_DIR    = Path(__file__).resolve().parents[1] / "data" / "hashes"

HASH_SOURCES = {
    "queries": [
        DATA_2409 / "queries_0_hash_values.xlsx",
        DATA_2409 / "queries_1_hash_values.xlsx",
    ],
    "references": [
        DATA_2409 / "references_0_hash_values.xlsx",
        DATA_2409 / "references_1_hash_values.xlsx",
    ],
}

HASH_COLS = ["filename", "aHash", "pHash", "hsvHash", "sHash"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_xlsx(xlsx_path: Path) -> list:
    """Copy xlsx to a temp file, parse it, return rows (including header)."""
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


def _load_and_merge(xlsx_paths: list[Path]) -> list[dict]:
    """Load one or more xlsx files and merge them into a list of dicts."""
    combined: dict[str, dict] = {}
    for xlsx_path in xlsx_paths:
        print(f"    Loading: {xlsx_path.name}")
        rows = _safe_load_xlsx(xlsx_path)
        header = rows[0]
        for row in rows[1:]:
            record = dict(zip(header, row))
            filename = str(record.get("filename", "")).strip()
            if not filename:
                continue
            combined[filename] = {
                "filename": filename,
                "aHash":    str(record.get("aHash",    "")).strip(),
                "pHash":    str(record.get("pHash",    "")).strip(),
                "hsvHash":  str(record.get("hsvHash",  "")).strip(),
                "sHash":    str(record.get("sHash",    "")).strip(),
            }
    return sorted(combined.values(), key=lambda x: x["filename"])


def _save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HASH_COLS)
        writer.writeheader()
        writer.writerows(records)
    print(f"    → Saved {len(records)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_hashes():
    print("=" * 60)
    print("STEP 3 - Hash Extraction from 2409_data")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for kind, paths in HASH_SOURCES.items():
        print(f"\n[{kind}]")
        records = _load_and_merge(paths)
        out_path = OUT_DIR / f"{kind}_hashes.csv"
        _save_csv(records, out_path)

    print("\n[PASS]  Hash extraction complete.")
    print("=" * 60)


if __name__ == "__main__":
    extract_hashes()
