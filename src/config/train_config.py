"""
train_config.py
===============
Centralised configuration for the XGBoost training pipeline.
All paths, random seeds, split ratios, and BK-Tree parameters live here
so that every stage of the pipeline draws from a single source of truth.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")

# Original image folders (3 collections × 500 images each = 1 500 total)
COLLECTIONS_DIRS = {
    "Azuki":                BASE_DIR / "Data_sets" / "Images_Azuki",
    "Bored_Ape_Yacht_Club": BASE_DIR / "Data_sets" / "Images_Bored_Ape_Yacht_Club",
    "Lil_Pudgys":           BASE_DIR / "Data_sets" / "Images_Lil_Pudgys",
}

# Duplications folder and metadata
DUPLICATIONS_DIR = BASE_DIR / "dataset_duplications"
DUPLICATIONS_METADATA_CSV = BASE_DIR / "duplications_metadata.csv"
ORIGINALS_METADATA_CSV    = BASE_DIR / "originals_metadata_with_hashes.csv"

# Output directories — V1 (distance-only features)
DATA_MANIFESTS_DIR = BASE_DIR / "data_manifests"
BKTREES_DIR        = BASE_DIR / "bktrees"
FEATURES_DIR       = BASE_DIR / "features"
MODELS_DIR         = BASE_DIR / "models"
REPORTS_DIR        = BASE_DIR / "reports"

# Output directories — V2 (distance + rank features)
FEATURES_V2_DIR    = BASE_DIR / "features_v2"
MODELS_V2_DIR      = BASE_DIR / "models_v2"
REPORTS_V2_DIR     = BASE_DIR / "reports_v2"

# Sibling project (for importing BKTree & distance functions)
SIBLING_PROJECT_DIR = BASE_DIR.parent / "NFT_program"
BK_TREE_MODULE_DIR  = SIBLING_PROJECT_DIR / "NFT-Duplications-detection-using-Image-hash-functions-main"

# ──────────────────────────────────────────────
# Sampling quantities (Steps 0 - 2)
# ──────────────────────────────────────────────
N_BASE_IMAGES    = 800   # Step 0: originals inserted into BK-Trees
N_ORIGINALS      = 700   # Step 1: remaining originals (negatives)
N_FAKES          = 700   # Step 2: fakes whose reference in base set

# ──────────────────────────────────────────────
# Split ratio (Step 3)
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.80       # 80 % train, 20 % test

# ──────────────────────────────────────────────
# BK-Tree / candidate retrieval (Steps 4 - 5)
# ──────────────────────────────────────────────
K_NEAREST = 50           # Top-K neighbours per hash tree
HASH_FUNCTIONS = ["aHash", "pHash", "sHash", "hsvHash"]

# Distance threshold used only during BK-Tree construction
# (large value so that top_k works without artificial cut-off)
BK_TREE_DISTANCE_THRESHOLD = 100

# ──────────────────────────────────────────────
# XGBoost training (Step 6)
# ──────────────────────────────────────────────
XGBOOST_PARAMS = {
    "max_depth":       6,
    "n_estimators":    300,
    "learning_rate":   0.1,
    "eval_metric":     "aucpr",
    "use_label_encoder": False,
    "random_state":    42,
}

# Grid for hyperparameter tuning (used with --tune flag)
# 5 × 4 × 4 = 80 combinations × 5 CV folds = 400 fits per version
XGBOOST_TUNE_GRID = {
    "max_depth":     [3, 4, 5, 6, 8],
    "n_estimators":  [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Feature column names
# ──────────────────────────────────────────────

# V1: 4 distance features only
FEATURE_COLUMNS_V1 = ["dist_aHash", "dist_pHash", "dist_sHash", "dist_hsvHash"]

# V2: 4 distances + 4 ranks + 2 summary features = 10 features
FEATURE_COLUMNS_V2 = [
    "dist_aHash", "dist_pHash", "dist_sHash", "dist_hsvHash",
    "rank_aHash", "rank_pHash", "rank_sHash", "rank_hsvHash",
    "n_trees_found", "best_rank",
]

# Default (backwards compatible)
FEATURE_COLUMNS = FEATURE_COLUMNS_V1
LABEL_COLUMN    = "label"


# ──────────────────────────────────────────────
# Version-aware config helper
# ──────────────────────────────────────────────

def get_version_config(version: str) -> dict:
    """Return paths and feature columns for a given model version ('v1' or 'v2')."""
    if version == "v1":
        return {
            "features_dir":    FEATURES_DIR,
            "models_dir":      MODELS_DIR,
            "reports_dir":     REPORTS_DIR,
            "feature_columns": FEATURE_COLUMNS_V1,
            "model_filename":  "fuzzy_hash_xgb.pkl",
        }
    elif version == "v2":
        return {
            "features_dir":    FEATURES_V2_DIR,
            "models_dir":      MODELS_V2_DIR,
            "reports_dir":     REPORTS_V2_DIR,
            "feature_columns": FEATURE_COLUMNS_V2,
            "model_filename":  "fuzzy_hash_xgb_v2.pkl",
        }
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'.")

