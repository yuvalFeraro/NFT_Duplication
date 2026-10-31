# NFT Duplication Detection — Presentation Brief
> Give this file to Claude and ask it to create a slide-by-slide presentation.

---

## Project Goal

Detect whether a given NFT image is a **near-duplicate (fake)** of an original NFT,
using perceptual hashing and a machine learning classifier.

The system is designed to scale to large NFT collections without computing
all pairwise distances (which would be O(n²)).

---

## Dataset

- **3 NFT collections** of original images:
  - Azuki (500 images)
  - Bored Ape Yacht Club (500 images)
  - Lil Pudgys (500 images)
  - **Total: 1,500 originals**

- **Fake (duplicated) images** stored in `dataset_duplications/`, organized into 4 categories:
  1. `1_geometric` — rotations, flips, crops
  2. `2_color` — hue shifts, saturation, brightness changes
  3. `3_noise_compression` — JPEG artifacts, blur, noise
  4. `4_content_modifications` — watermarks, overlays, text

Each fake image has a known **reference image** (the original it was derived from).

---

## Hash Functions Used

The system uses **4 perceptual hash functions**, each capturing different visual properties:

| Hash | What it captures |
|------|-----------------|
| `aHash` (Average Hash) | Global brightness structure |
| `pHash` (Perceptual Hash) | DCT frequency content |
| `sHash` (Segmentation Hash) | Local segment-level similarity (ImageMultiHash) |
| `hsvHash` (Color Hash) | Color distribution (HSV histogram) |

Each hash produces a compact binary fingerprint. Similarity is measured via **Hamming distance** (number of differing bits). A distance of 0 = identical; higher = more different.

---

## Core Idea: BK-Tree for Efficient Retrieval

A **BK-Tree** (Burkhard-Keller Tree) is a metric tree built for discrete distance functions (like Hamming distance). It allows fast retrieval of the **top-K nearest neighbors** without scanning the entire collection.

- **4 BK-Trees are built** — one per hash function.
- Each tree holds all base (known) images inserted by their hash fingerprint.
- To query a new image: compute its 4 hashes → query all 4 trees → take the **union** of top-50 candidates from each tree.
- Result: up to ~200 unique candidate images to inspect.

This reduces the comparison cost from O(n) to O(log n) per query.

---

## Machine Learning: XGBoost Classifier

For each (query image, candidate image) pair returned by the BK-Trees, a **4-dimensional feature vector** is computed:

```
[dist_aHash,  dist_pHash,  dist_sHash,  dist_hsvHash]
```

These are the Hamming distances between the query and the candidate in each hash space.

An **XGBoost binary classifier** is trained to predict:
- **Label 1** (duplicate): the candidate IS the true reference of a known fake
- **Label 0** (non-duplicate): the candidate is just a visually similar but unrelated image

**V2 model** additionally uses 6 rank-based features (10 features total):
```
[dist_aHash, dist_pHash, dist_sHash, dist_hsvHash,
 rank_aHash, rank_pHash, rank_sHash, rank_hsvHash,
 n_trees_found, best_rank]
```

---

## Training Pipeline — Step by Step

### Step 0: Image Selection
**Script:** `src/data_generation/select_images.py`

Select images for training/testing:
- **800 base images** — originals that each have at least one fake in the dataset.
  These are inserted into the BK-Trees (they form the "known collection").
- **700 remaining originals** — originals with no fakes; used as negative query images.
- **700 fake images** — whose reference image is among the 800 base images.

### Step 1: 80/20 Train/Test Split
**Script:** `src/data_generation/select_images.py` (same script, Step 3 inside)

Split is done carefully to **prevent data leakage**:
1. Split the 800 base images 80/20 → 640 train-base / 160 test-base
2. Each fake is assigned to whichever split its reference image landed in
   (so a fake's reference is always in the same split's BK-Tree)
3. Split the 700 originals 80/20 independently

Result: 6 CSV manifest files in `data_manifests/`:
- `train_base.csv`, `test_base.csv`
- `train_originals.csv`, `test_originals.csv`
- `train_fakes.csv`, `test_fakes.csv`

### Step 2: Build BK-Trees & Extract Features
**Script:** `src/feature_extraction/generate_features.py`

**For each split (train and test separately):**

1. **Build 4 BK-Trees** from the base images of that split (640 for train, 160 for test).
2. For each **query image** (originals + fakes in that split):
   - Compute its 4 hash values
   - Query all 4 BK-Trees → take union of top-50 candidates per tree
   - For each candidate, compute the 4 Hamming distances → one feature row
   - **Label = 1** if the candidate is the true reference of this fake, **Label = 0** otherwise
3. Save all rows to `train_features.csv` / `test_features.csv`

**Key design — Hard Negative Mining:**
The negative examples (label=0) come from BK-Tree results — i.e. images that are already
visually similar but are NOT the true reference. These are the hardest possible negatives,
matching the exact deployment scenario.

### Step 3: Hyperparameter Tuning + Model Training
**Script:** `src/model/train_xgboost.py`

1. **GridSearchCV** over 80 parameter combinations × 5 CV folds = **400 model fits**:
   - `max_depth`: [3, 4, 5, 6, 8]
   - `n_estimators`: [100, 200, 300, 500]
   - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
   - Scoring metric: **F1** (optimized for the positive/duplicate class)

2. Best parameters are selected and used to **train the final model on 100% of train data**.

3. A 5-fold stratified CV is run with the best params for **evaluation reporting**.

4. Class imbalance is handled via `scale_pos_weight = n_negatives / n_positives`.

5. Model saved to `models/fuzzy_hash_xgb.pkl` (V1) or `models_v2/fuzzy_hash_xgb_v2.pkl` (V2).

### Step 4: Evaluation on Test Set
**Script:** `src/model/evaluate.py`

1. Load `test_features.csv` and the saved model.
2. **Per-pair metrics**: ROC-AUC, PR-AUC on all (query, candidate) pairs.
3. **Query-level (image-level) metrics** — the real deployment metric:
   - For each query image: `max_prob = max(probabilities of all its candidates)`
   - If `max_prob > threshold` → predict FAKE
4. **Threshold sweep** from 0.30 to 0.95 in steps of 0.05:
   - Reports Recall, Precision, FPR, F1 at each threshold
   - Selects best threshold by F1
   - Also reports best threshold subject to FPR ≤ 5%
5. Best threshold is saved back into the model file for deployment.

---

## Deployment Logic (Inference on a New Image)

When a new unknown image arrives:

```
1. Compute its 4 hash values (aHash, pHash, sHash, hsvHash)
2. Query all 4 BK-Trees (built from the full known collection)
   → get union of top-50 candidates per tree (~200 candidates)
3. For each candidate:
   → compute [dist_aHash, dist_pHash, dist_sHash, dist_hsvHash]
   → XGBoost.predict_proba() → probability score
4. max_prob = max of all candidate probabilities
5. If max_prob > threshold → FLAG AS DUPLICATE
   If max_prob <= threshold → AUTHENTIC
```

---

## Two Model Versions

| | V1 | V2 |
|---|---|---|
| Features | 4 (distances only) | 10 (distances + ranks) |
| Feature set | dist_aHash, dist_pHash, dist_sHash, dist_hsvHash | V1 features + rank_aHash, rank_pHash, rank_sHash, rank_hsvHash, n_trees_found, best_rank |
| Model file | `fuzzy_hash_xgb.pkl` | `fuzzy_hash_xgb_v2.pkl` |

---

## Key Design Decisions

1. **Separate BK-Trees for train and test** — prevents any information from the training base leaking into test evaluation.

2. **Hard negative mining** — negatives come from BK-Tree results (already visually close images), not random pairs. This exactly matches the deployment scenario.

3. **Fake → reference always in same split** — when splitting base images 80/20, all fakes follow their reference. A fake is never evaluated against a tree that doesn't contain its reference.

4. **Max-aggregation at query level** — one suspicious candidate is enough to flag an image. This is the correct aggregation for the "does this image have ANY match in the collection?" question.

5. **Threshold selected on test set** — the threshold is a deployment decision (recall vs. precision trade-off), not a model parameter. It is selected on held-out data to reflect real-world performance honestly.

---

## Command-Line Usage

```bash
# Full pipeline — V1
python src/data_generation/select_images.py
python src/feature_extraction/generate_features.py
python src/model/train_xgboost.py
python src/model/evaluate.py

# Full pipeline — V2
python src/data_generation/select_images.py
python src/feature_extraction/generate_features.py --version v2
python src/model/train_xgboost.py --version v2
python src/model/evaluate.py --version v2
```

---

## Suggested Slide Structure for Claude

1. **Title slide** — "NFT Duplication Detection Using Perceptual Hashing + XGBoost"
2. **Problem statement** — What is an NFT fake? Why is detection hard at scale?
3. **Dataset** — 3 collections, 1500 originals, 4 types of fakes
4. **Perceptual hashing** — What are the 4 hashes and what do they capture?
5. **BK-Tree** — How it enables fast nearest-neighbor retrieval at scale
6. **Pipeline overview diagram** — All 4 steps in one visual
7. **Data selection & split** — 800 base / 700 originals / 700 fakes, 80/20 split, leakage prevention
8. **Feature extraction** — BK-Tree query → 50 candidates → distance vectors → labels
9. **Hard negative mining** — Why visually-similar non-duplicates are the best negatives
10. **Model training** — XGBoost, GridSearchCV, scale_pos_weight, 400 fits
11. **Two model versions** — V1 vs V2 features table
12. **Evaluation** — Per-pair vs. query-level metrics, threshold sweep
13. **Deployment inference flow** — Step-by-step diagram of how a new image is classified
14. **Results / conclusions** (add your actual numbers here)
