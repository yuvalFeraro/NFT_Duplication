# NFT_program_V1 — DISC-21 Hash-Threshold Detector

## Overview
This pipeline trains and evaluates a **Minimal-Distance Detector** on the DISC-21 dataset using four perceptual hashes (aHash, pHash, hsvHash, sHash) with a brute-force threshold search.

Three detector variants are evaluated:
- **k=1** — duplicate if **any** hash distance < its threshold (OR logic)
- **k=2** — duplicate if **≥ 2** hashes have distance < threshold
- **k=3** — duplicate if **≥ 3** hashes have distance < threshold

**Objective:** Maximise Recall subject to FPR ≤ 20%

---

## Dataset

| Folder | Contents |
|---|---|
| `Data_sets/DISC-21/` | 1032 reference images (R*.jpg) |
| `dataset_duplications/DISC21_duplications/` | 516 query/duplication images (Q*.jpg) |
| `data_sets/DISC-21/2409_data/` | Pre-computed hash xlsx files |

- 516 references **have** a paired query → split into `base` + `duplications`
- 516 references have **no** query pair → become `negatives`

---

## Directory Structure

```
NFT_program_V1/
├── run_all.py                     ← entry point
├── scripts/
│   ├── step1_verify_data.py       ← verify all query–reference pairs exist
│   ├── step2_split_data.py        ← 80/20 train/test split (coupled)
│   ├── step3_extract_hashes.py    ← build clean hash CSVs
│   └── step4_train_and_test.py    ← brute-force sweep + test
├── data/
│   ├── splits/                    ← train/test CSV manifests
│   └── hashes/                    ← references_hashes.csv, queries_hashes.csv
└── results/
    ├── train_results_k1.csv
    ├── train_results_k2.csv
    ├── train_results_k3.csv
    └── test_report.txt
```

---

## How to Run

```bash
# Run all 4 steps
python run_all.py

# Or run individual steps
python run_all.py --steps 1
python run_all.py --steps 2 3
python run_all.py --steps 4
```

Steps must be run in order on first use (step 2 & 3 produce files that step 4 needs).

---

## Metrics

| Metric | Description |
|---|---|
| Recall | TP / (TP + FN) — fraction of duplicates correctly detected |
| FPR | FP / (FP + TN) — fraction of negatives incorrectly flagged |
| Precision | TP / (TP + FP) |
| F1 | Harmonic mean of Precision and Recall |

Training shows Recall + FPR.  
Testing shows all four metrics.

---

## sHash Distance
Uses the same logic as `minimal_distance_detector.py`:
```python
_, total_diff = hash1.hash_diff(hash2, hamming_cutoff=100)
dist = total_diff / len(hash1.segment_hashes)
```
This is the average per-segment Hamming distance.
