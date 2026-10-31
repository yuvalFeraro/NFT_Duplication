"""
extract_hashes.py
=================
Modular feature extraction pipeline for computing perceptual hashes (aHash, pHash, sHash, hsvHash)
for NFT image datasets.

Follows the schema of duplications_metadata and project guidelines in PROJECT_RULES.md.
Reusable for any future datasets or re-runs.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from PIL import Image
import imagehash


# Default base directory
BASE_DIR = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")

COLLECTIONS_DIRS = {
    "Azuki": BASE_DIR / "Data_sets" / "Images_Azuki",
    "Bored_Ape_Yacht_Club": BASE_DIR / "Data_sets" / "Images_Bored_Ape_Yacht_Club",
    "Lil_Pudgys": BASE_DIR / "Data_sets" / "Images_Lil_Pudgys"
}

DEFAULT_OUTPUT_EXCEL = BASE_DIR / "originals_metadata_with_hashes.xlsx"
DEFAULT_OUTPUT_CSV = BASE_DIR / "originals_metadata_with_hashes.csv"

DUPLICATIONS_BASE_DIR = BASE_DIR / "dataset_duplications"
DUPLICATIONS_METADATA_EXCEL = BASE_DIR / "duplications_metadata.xlsx"
DUPLICATIONS_METADATA_CSV = BASE_DIR / "duplications_metadata.csv"

MASTER_COMBINED_EXCEL = BASE_DIR / "all_images_metadata_with_hashes.xlsx"
MASTER_COMBINED_CSV = BASE_DIR / "all_images_metadata_with_hashes.csv"


def compute_hashes_for_file(file_path_str: str) -> Optional[Dict[str, str]]:
    """Worker function: loads image, converts to RGB, and computes 4 perceptual hashes."""
    try:
        file_path = Path(file_path_str)
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            ahash = str(imagehash.average_hash(img_rgb))
            phash = str(imagehash.phash(img_rgb))
            hsvhash = str(imagehash.colorhash(img_rgb))
            shash = str(imagehash.crop_resistant_hash(img_rgb))
            
            return {
                "filename": file_path.name,
                "aHash": ahash,
                "sHash": shash,
                "pHash": phash,
                "hsvHash": hsvhash
            }
    except Exception as e:
        print(f"Error computing hashes for {file_path_str}: {e}")
        return None


def extract_original_images_metadata(
    collections_dirs: Dict[str, Path] = COLLECTIONS_DIRS,
    output_excel: Path = DEFAULT_OUTPUT_EXCEL,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Scans the original collection folders, calculates 4 perceptual hashes,
    and constructs a DataFrame matching the duplications_metadata schema.
    """
    start_time = time.time()
    records: List[Dict[str, Any]] = []
    task_args = []

    print("==================================================")
    print("Perceptual Hash Extraction for Original Images")
    print("==================================================")

    # 1. Collect all image file paths and assign metadata
    for col_name, col_dir in collections_dirs.items():
        if not col_dir.exists():
            print(f"Warning: Directory not found: {col_dir}")
            continue
        
        files = sorted(list(col_dir.glob("*.png")))
        print(f"Found {len(files)} original images in {col_name}")
        for f in files:
            metadata = {
                "filename": f.name,
                "ref_image": f.name,
                "type": "original",
                "sub_type": "original",
                "collection": col_name,
                "parameters": "{}",
                "is_duplicate": 0
            }
            records.append(metadata)
            task_args.append((str(f), metadata))

    total_images = len(task_args)
    print(f"\nTotal original images to process: {total_images}")
    print(f"Extracting aHash, sHash, pHash, hsvHash across {max_workers} processes...")

    # 2. Parallel processing using ProcessPoolExecutor for CPU-bound hashing
    hash_lookup: Dict[str, Dict[str, str]] = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_hashes_for_file, f_str): f_str for f_str, _ in task_args}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                hash_lookup[res["filename"]] = res
            completed_count += 1
            
            if completed_count % 100 == 0 or completed_count == total_images:
                elapsed = time.time() - start_time
                fps = completed_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed_count}/{total_images} images processed ({fps:.1f} images/sec)...")

    # 3. Assemble combined DataFrame with consistent columns
    final_rows: List[Dict[str, Any]] = []
    for meta in records:
        f_name = meta["filename"]
        h_info = hash_lookup.get(f_name, {})
        final_rows.append({
            "filename": meta["filename"],
            "ref_image": meta["ref_image"],
            "type": meta["type"],
            "sub_type": meta["sub_type"],
            "collection": meta["collection"],
            "parameters": meta["parameters"],
            "is_duplicate": meta["is_duplicate"],
            "aHash": h_info.get("aHash", ""),
            "sHash": h_info.get("sHash", ""),
            "pHash": h_info.get("pHash", ""),
            "hsvHash": h_info.get("hsvHash", "")
        })

    df = pd.DataFrame(final_rows)

    # 4. Save to Excel and CSV
    print(f"\nSaving results to:\n  - {output_excel}\n  - {output_csv}")
    df.to_excel(output_excel, index=False)
    df.to_csv(output_csv, index=False)

    total_time = time.time() - start_time
    print(f"Extraction completed in {total_time:.2f} seconds ({len(df)} records).")
    return df


def extract_duplicates_metadata_hashes(
    metadata_path: Path = DUPLICATIONS_METADATA_EXCEL,
    duplications_dir: Path = DUPLICATIONS_BASE_DIR,
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Calculates 4 perceptual hashes for all 2,000 synthetic duplicate images
    and updates duplications_metadata.xlsx and CSV with the hash columns.
    """
    start_time = time.time()
    print("==================================================")
    print("Perceptual Hash Extraction for Duplicate Images")
    print("==================================================")

    if not metadata_path.exists():
        metadata_path = duplications_dir / "duplications_metadata.xlsx"
    if not metadata_path.exists():
        metadata_path = duplications_dir / "duplications_metadata.csv"
        
    print(f"Loading metadata from: {metadata_path}")
    if metadata_path.suffix == ".csv":
        df = pd.read_csv(metadata_path)
    else:
        df = pd.read_excel(metadata_path)
        
    print(f"Found {len(df)} duplicate records in metadata.")

    # Locate each image file in the subfolders
    category_subfolders = [
        duplications_dir / "1_geometric",
        duplications_dir / "2_color",
        duplications_dir / "3_noise_compression",
        duplications_dir / "4_content_modifications"
    ]

    file_paths: List[str] = []
    missing_count = 0

    for filename in df["filename"]:
        found_path = None
        for folder in category_subfolders:
            candidate = folder / filename
            if candidate.exists():
                found_path = str(candidate)
                break
        if found_path:
            file_paths.append(found_path)
        else:
            missing_count += 1
            file_paths.append("")
            print(f"Warning: could not locate duplicate image: {filename}")

    if missing_count > 0:
        print(f"Warning: {missing_count} duplicate files could not be found.")

    total_files = len([p for p in file_paths if p])
    print(f"Computing aHash, sHash, pHash, hsvHash across {max_workers} processes for {total_files} files...")

    hash_lookup: Dict[str, Dict[str, str]] = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_hashes_for_file, p): p for p in file_paths if p}

        # updates the hash lookup with the computed hashes
        # future is an object of the type Future, not the dict
        for future in as_completed(futures):
            res = future.result()
            if res:
                hash_lookup[res["filename"]] = res
            completed_count += 1

            if completed_count % 100 == 0 or completed_count == total_files:
                elapsed = time.time() - start_time
                fps = completed_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed_count}/{total_files} duplicates hashed ({fps:.1f} images/sec)...")

    # Map hashes back to DataFrame columns
    df["aHash"] = df["filename"].apply(lambda fn: hash_lookup.get(fn, {}).get("aHash", ""))
    df["sHash"] = df["filename"].apply(lambda fn: hash_lookup.get(fn, {}).get("sHash", ""))
    df["pHash"] = df["filename"].apply(lambda fn: hash_lookup.get(fn, {}).get("pHash", ""))
    df["hsvHash"] = df["filename"].apply(lambda fn: hash_lookup.get(fn, {}).get("hsvHash", ""))

    # Save to both locations (root and dataset_duplications/)
    save_targets = [
        BASE_DIR / "duplications_metadata.xlsx",
        BASE_DIR / "duplications_metadata.csv",
        duplications_dir / "duplications_metadata.xlsx",
        duplications_dir / "duplications_metadata.csv"
    ]

    print("\nSaving updated duplicates metadata with hashes to:")
    for target in save_targets:
        print(f"  - {target}")
        if target.suffix == ".csv":
            df.to_csv(target, index=False)
        else:
            df.to_excel(target, index=False)

    total_time = time.time() - start_time
    print(f"Duplicates hash extraction completed in {total_time:.2f} seconds ({len(df)} records).")
    return df


def create_master_combined_dataset(
    originals_excel: Path = DEFAULT_OUTPUT_EXCEL,
    duplicates_excel: Path = DUPLICATIONS_METADATA_EXCEL,
    output_excel: Path = MASTER_COMBINED_EXCEL,
    output_csv: Path = MASTER_COMBINED_CSV
) -> pd.DataFrame:
    """
    Concatenates original images (1,500) and duplicates (2,000) into a single master
    dataset containing all 3,500 images with unified schema and all 4 hashes.
    """
    print("\n==================================================")
    print("Creating Master Combined Dataset (3,500 Images)")
    print("==================================================")

    if not originals_excel.exists():
        raise FileNotFoundError(f"Originals metadata file not found: {originals_excel}")
    if not duplicates_excel.exists():
        duplicates_excel = DUPLICATIONS_BASE_DIR / "duplications_metadata.xlsx"

    orig_df = pd.read_excel(originals_excel)
    dup_df = pd.read_excel(duplicates_excel)

    print(f"Loaded {len(orig_df)} originals (is_duplicate=0)")
    print(f"Loaded {len(dup_df)} duplicates (is_duplicate=1)")

    # Ensure identical column order
    cols = [
        "filename", "ref_image", "type", "sub_type", "collection",
        "parameters", "is_duplicate", "aHash", "sHash", "pHash", "hsvHash"
    ]

    orig_df = orig_df[cols]
    dup_df = dup_df[cols]

    master_df = pd.concat([orig_df, dup_df], ignore_index=True)
    print(f"Combined total: {len(master_df)} rows")

    print(f"Saving master dataset to:\n  - {output_excel}\n  - {output_csv}")
    master_df.to_excel(output_excel, index=False)
    master_df.to_csv(output_csv, index=False)
    print("Master dataset saved successfully!")
    return master_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract perceptual hashes for NFT datasets.")
    parser.add_argument(
        "--mode",
        choices=["originals", "duplicates", "all"],
        default="duplicates",
        help="Extraction mode: originals (1,500), duplicates (2,000), or all (3,500)"
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    if args.mode in ["duplicates", "all"]:
        extract_duplicates_metadata_hashes(max_workers=args.workers)
        create_master_combined_dataset()
    elif args.mode == "originals":
        extract_original_images_metadata(max_workers=args.workers)

