# Synthetic NFT Duplication Generation Plan

## 1. Overview & Objective
This plan outlines the methodology for constructing a controlled, diverse, and reproducible benchmark of **2,000 synthetic duplicate NFT images** (500 for each of the 4 manipulation categories). 

All manipulations will use randomized parameters (angles, scale factors, positions, text content, noise distributions) to ensure wide coverage, challenging both perceptual hashing algorithms (`aHash`, `pHash`, `sHash`, `hsvHash`) and the downstream classifier (`XGBoost`).

---

## 2. Source Image Distribution
Images are sampled uniformly across the three curated original collections located in `NFT_program_V2`:
* **Azuki** ([Images_Azuki](file:///c:/Users/ferar/OneDrive%20-%20Technion/NFT_Duplication_Prj/NFT_program_V2/Images_Azuki)): 500 images
* **Bored Ape Yacht Club** ([Images_Bored_Ape_Yacht_Club](file:///c:/Users/ferar/OneDrive%20-%20Technion/NFT_Duplication_Prj/NFT_program_V2/Images_Bored_Ape_Yacht_Club)): 500 images
* **Lil Pudgys** ([Images_Lil_Pudgys](file:///c:/Users/ferar/OneDrive%20-%20Technion/NFT_Duplication_Prj/NFT_program_V2/Images_Lil_Pudgys)): 500 images

For each of the 4 categories (500 duplicates each):
* ~167 images sampled from Azuki
* ~167 images sampled from Bored Ape Yacht Club
* ~166 images sampled from Lil Pudgys
* **Total:** 500 duplicates per category $\times$ 4 categories = **2,000 duplicate images**.

---

## 3. Manipulation Categories & Randomization Specifications

### Category 1: Geometric Transformations (500 duplicates)
Sub-types randomly chosen per instance:
1. **Cropping (חיתוך)**:
   * Random crop margin: $5\%$ to $30\%$ removed from borders (random independent margins for left, top, right, bottom).
   * Resized back to original canvas dimensions using bicubic interpolation.
2. **Rotation (סיבוב)**:
   * Continuous subtle rotation: angle $\theta \sim \mathcal{U}(-15^\circ, +15^\circ)$ with automatic edge reflection/padding.
   * Cardinal rotation: $90^\circ, 180^\circ$, or $270^\circ$ (selected with 15% probability).
3. **Flipping (היפוך)**:
   * Horizontal mirror reflection (90% chance) or vertical flip (10% chance).
4. **Rescaling & Aspect Ratio Distortion (שינוי יחס ממדים)**:
   * Anisotropic stretching/compression on X or Y axis with random scale factor $s \sim \mathcal{U}(0.80, 1.25)$, then padded or resized back.

### Category 2: Color & Photometric Modifications (500 duplicates)
Sub-types randomly chosen per instance:
1. **Brightness & Contrast Adjustments**:
   * Brightness multiplier $\alpha \sim \mathcal{U}(0.60, 1.50)$.
   * Contrast multiplier $\beta \sim \mathcal{U}(0.60, 1.50)$.
2. **Hue & Saturation Shifts**:
   * Conversion to HSV color space.
   * Hue shift $\Delta H \sim \mathcal{U}(-35^\circ, +35^\circ)$.
   * Saturation multiplier $S \sim \mathcal{U}(0.50, 2.00)$.
3. **Grayscale Conversion**:
   * Conversion to single-channel luminance, replicated across RGB channels.
4. **Color Temperature / Tint Filter**:
   * Warm tint (boost red/yellow), cool tint (boost cyan/blue), or sepia transformation with randomized intensity factor $\gamma \sim \mathcal{U}(0.2, 0.6)$.

### Category 3: Noise & Compression Attacks (500 duplicates)
Sub-types randomly chosen per instance:
1. **Gaussian Noise**:
   * Additive zero-mean Gaussian noise with random standard deviation $\sigma \sim \mathcal{U}(10.0, 45.0)$, clipped to $[0, 255]$.
2. **Salt & Pepper Noise**:
   * Random impulse noise replacing $p \sim \mathcal{U}(0.5\%, 3.0\%)$ of pixels with 0 or 255.
3. **JPEG Compression Artifacts**:
   * Re-encoding to JPEG at quality factor $Q \sim \mathcal{U}(10, 55)$ (simulating lossy compression and re-saving).
4. **Blurring & Smoothing**:
   * Gaussian blur with random kernel radius $r \sim \mathcal{U}(1.0, 4.5)$, or median filtering.

### Category 4: Content Modifications & Overlays (500 duplicates)
Sub-types randomly chosen per instance:
1. **Watermark / Text Overlay**:
   * Text vocabulary: `"COPY"`, `"VERIFIED"`, `"SAMPLE"`, `"PROOF"`, `"NFT"`, `"AUTHENTIC"`, or random hash string.
   * Font size: $24\text{px}$ to $72\text{px}$ scaled relative to image resolution.
   * Position: Top-left, center, bottom-right, or diagonal center banner.
   * Opacity: Alpha channel $\alpha \sim \mathcal{U}(0.20, 0.75)$.
   * Color: White, black, red, or yellow.
2. **Borders & Banners**:
   * Decorative solid/patterned border added around edges (thickness $2\%$ to $8\%$ of width).
3. **Background Manipulation / Color Tinting**:
   * Background detection / corner color sampling with replacement by solid color or gradient background.

---

## 4. Directory Structure & Output Layout
Per Rule 1 and Rule 4 of the project guidelines, output directories and scripts are strictly structured:

```text
NFT_program_V2/
├── dataset_duplications/
│   ├── 1_geometric/                 (500 .png images)
│   ├── 2_color/                     (500 .png images)
│   ├── 3_noise_compression/         (500 .png images)
│   ├── 4_content_modifications/     (500 .png images)
│   └── duplications_manifest.xlsx   (Full ground truth dataset metadata)
└── src/
    └── data_generation/
        └── generate_duplications.py (Production generation pipeline)
```

---

## 5. Ground Truth Metadata Schema (`duplications_manifest.xlsx`)
Each generated image row in the manifest will record:
* `duplicate_filename`: (e.g., `Bored_Ape_Yacht_Club_25_geo_crop.png`)
* `original_filename`: (e.g., `Bored_Ape_Yacht_Club_25.png`)
* `collection`: (`Bored_Ape_Yacht_Club`, `Azuki`, `Lil_Pudgys`)
* `category`: (`Geometric`, `Color`, `Noise_Compression`, `Content`)
* `subtype`: (`Cropping`, `Rotation`, `Brightness`, `Gaussian_Noise`, `Watermark`, etc.)
* `parameters`: JSON string with exact parameters applied (e.g., `{"angle": 8.5, "fill": "reflect"}`)
* `is_duplicate`: `1` (binary ground-truth label)

---

## 6. Scientific Analysis Goals (Recall, FPR, & Scalability)
Once generated, this dataset enables rigorous evaluation:
* **Per-category Recall & FPR**: Identifying the precise vulnerability thresholds of `aHash`, `pHash`, `sHash`, and `hsvHash`.
* **Classifier Stress Testing**: Demonstrating the performance of the XGBoost classifier when distinguishing subtle vs. severe manipulations against the BK-Tree candidate index.
