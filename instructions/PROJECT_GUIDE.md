# NFT_program_V2 — Comprehensive Project Guide

Welcome to **NFT_program_V2**, a modular, high-performance system for detecting NFT image duplications and evaluating perceptual hash bounds. This document serves as the primary manual for the repository architecture, datasets, feature extraction routines, and replication procedures.

---

## 1. Project Directory Structure

```text
NFT_program_V2/
├── GEMINI.md                          # IDE / AI operational rules & guidelines
├── PROJECT_RULES.md                   # Project rules and clean-code constraints
├── PROJECT_GUIDE.md                   # This project guide & documentation
├── new_model_plan_v2.md               # Scientific research plan & theoretical bounds
├── duplication_generation_plan.md     # Synthetic duplication methodology
│
├── Images_Azuki/                      # 500 original images (Azuki_0.png .. 499.png)
├── Images_Bored_Ape_Yacht_Club/       # 500 original images (Bored_Ape_Yacht_Club_0.png .. 499.png)
├── Images_Lil_Pudgys/                 # 500 original images (Lil_Pudgys_0.png .. 499.png)
│
├── dataset_duplications/              # 2,000 synthetic duplicate images
│   ├── 1_geometric/                  # 500 images (cropping, rotation, flipping, aspect ratio)
│   ├── 2_color/                      # 500 images (brightness, contrast, hue, saturation, grayscale, tint)
│   ├── 3_noise_compression/          # 500 images (gaussian noise, salt & pepper, jpeg artifacts, blur)
│   ├── 4_content_modifications/      # 500 images (watermarks, borders, background tint)
│   ├── duplications_metadata.xlsx    # Ground-truth metadata sheet for duplicates
│   └── duplications_metadata.csv     # CSV version
│
├── duplications_metadata.xlsx         # 2,000 duplicate images + 4 perceptual hashes
├── duplications_metadata.csv          # CSV version
├── originals_metadata_with_hashes.xlsx# 1,500 original images + 4 perceptual hashes
├── originals_metadata_with_hashes.csv # CSV version
├── all_images_metadata_with_hashes.xlsx# Master dataset: 3,500 images (originals + duplicates) + all hashes
├── all_images_metadata_with_hashes.csv # CSV version
│
└── src/                               # Production modular codebase
    ├── data_generation/
    │   └── generate_duplications.py   # Synthetic duplicate generator
    └── feature_extraction/
        └── extract_hashes.py          # Multiprocess perceptual hash extraction pipeline
```

---

## 2. Dataset Overview

The dataset consists of **3,500 total images** partitioned into original reference images and synthetic duplicates:

| Dataset Subset | Image Count | Resolution / Style | Ground Truth `is_duplicate` |
| :--- | :--- | :--- | :--- |
| **Azuki Originals** | 500 | High-res (~1.2 MB / image), Anime art style | `0` |
| **Bored Ape Originals** | 500 | Medium-res (~130 KB / image), Illustrative vector PFP | `0` |
| **Lil Pudgys Originals** | 500 | Medium-res (~240 KB / image), 2D Cartoon vector PFP | `0` |
| **Synthetic Duplications** | 2,000 (500 per class) | Generated across 4 distinct manipulation types | `1` |

---

## 3. Metadata Schemas

Both the duplicates and original metadata files share a consistent, unified schema:

| Column | Type | Description | Original Image Example | Duplicate Image Example |
| :--- | :--- | :--- | :--- | :--- |
| `filename` | string | Name of the file | `Azuki_0.png` | `Azuki_200_dup_geo_6.png` |
| `ref_image` | string | Source original image filename | `Azuki_0.png` | `Azuki_200.png` |
| `type` | string | Broad manipulation category | `original` | `geometric transformation` |
| `sub_type` | string | Fine-grained transformation type | `original` | `rotation` |
| `collection` | string | Source collection | `Azuki` | `Azuki` |
| `parameters` | JSON string | Exact randomization parameters applied | `{}` | `{"angle_deg": 90}` |
| `is_duplicate`| int (0 or 1) | Binary classification label | `0` | `1` |
| `aHash` | hex string | Average perceptual hash (64-bit) | `ff818181...` | Computed hash |
| `sHash` | hex string | Crop-resistant hash | `...` | Computed hash |
| `pHash` | hex string | Discrete Cosine Transform (DCT) perceptual hash | `e3c281...` | Computed hash |
| `hsvHash` | hex string | Color HSV palette hash | `001f...` | Computed hash |

---

## 4. Perceptual Hash Functions & Characteristics

The system relies on 4 complementary hashing algorithms from the `imagehash` library to evaluate duplication:

1. **`aHash` (`imagehash.average_hash`)**:
   * **Mechanism**: Downscales to $8 \times 8$ grayscale, computes average luminance, outputs 64-bit threshold bitmap.
   * **Strengths**: Extremely fast; resilient to mild noise, minor contrast shifts, and format compression.
   * **Weaknesses**: Sensitive to cropping, shifting, and rotation.

2. **`pHash` (`imagehash.phash`)**:
   * **Mechanism**: Computes $32 \times 32$ Discrete Cosine Transform (DCT) and extracts low-frequency coefficients.
   * **Strengths**: Standard in industry; resilient to scaling, compression artifacts, and color balance.
   * **Weaknesses**: Vulnerable to large crops and significant geometric rotation.

3. **`sHash` (`imagehash.crop_resistant_hash`)**:
   * **Mechanism**: Multi-scale feature point identification resilient to bounding box changes.
   * **Strengths**: Specifically engineered to detect cropped portions and localized alterations.
   * **Weaknesses**: Computationally heavier (takes ~0.15s per image).

4. **`hsvHash` (`imagehash.colorhash`)**:
   * **Mechanism**: Samples dominant HSV color distribution.
   * **Strengths**: Highly sensitive to color palette modifications; agnostic to geometric distortions.
   * **Weaknesses**: Does not capture spatial structures.

---

## 5. Reusable Execution Commands

All scripts are stored inside `src/` to prevent root clutter and enable clean reproducibility:

### A. Extracting Hashes for Original Images
To extract perceptual hashes for all original images across the 3 folders:
```bash
python src/feature_extraction/extract_hashes.py --mode originals
```
* **Output**: Writes `originals_metadata_with_hashes.xlsx` and `originals_metadata_with_hashes.csv`.

### B. Extracting Hashes for Duplicate Images & Master Dataset
To extract perceptual hashes for all 2,000 duplicate images and build the 3,500 master dataset:
```bash
python src/feature_extraction/extract_hashes.py --mode duplicates
```
* **Output**: Updates `duplications_metadata.xlsx` and generates `all_images_metadata_with_hashes.xlsx`.

### C. Extracting Hashes for Everything (All 3,500 Images)
```bash
python src/feature_extraction/extract_hashes.py --mode all
```

### D. Generating New Synthetic Duplicates
To regenerate or extend the 2,000 synthetic duplicates:
```bash
python src/data_generation/generate_duplications.py
```
* **Output**: Populates `dataset_duplications/` and writes `duplications_metadata.xlsx`.

### C. Processing Any New Collection in the Future
To extract hashes for a new incoming collection (e.g., `Images_New_Collection/`), simply call the function programmatically:
```python
from pathlib import Path
from src.feature_extraction.extract_hashes import extract_original_images_metadata

new_collections = {
    "NewCollection": Path(r"path/to/Images_New_Collection")
}

df = extract_original_images_metadata(
    collections_dirs=new_collections,
    output_excel=Path("new_collection_hashes.xlsx"),
    max_workers=8
)
```

---

## 6. Project Guidelines Summary
* **No root scripts**: Keep `.py` scripts inside `src/` subdirectories.
* **Inline commands for inspections**: Run `python -c "..."` instead of throwaway check scripts.
* **Update existing code**: Avoid creating duplicate version names like `_v2`, `_2409`.
* **Always verify trade-offs**: Benchmark models against Scalability ($O(\log N)$ retrieval vs $O(N)$) and Recall vs. False Positive Rate (FPR).
