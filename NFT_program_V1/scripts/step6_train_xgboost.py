"""
step6_train_xgboost.py
======================
Train an XGBoost binary classifier on DISC-21 V2 features (10 features:
4 Hamming distances + 4 ranks + 2 summary features).

Pipeline:
  1. Load train_features.csv produced by step5_generate_features.py.
  2. Run GridSearchCV (5-fold stratified, scoring=F1) to find the best
     hyperparameters for XGBoost.
  3. Retrain the final model with the winning params on the full training set.
  4. Evaluate on the held-out test_features.csv.
  5. Save: model (.pkl), tuning_report.md, training_report.md, test_report.md

Feature vector (V2, 10 features):
    dist_aHash, dist_pHash, dist_hsvHash, dist_sHash
    rank_aHash, rank_pHash, rank_hsvHash, rank_sHash
    n_trees_found, best_rank

Run:
    python step6_train_xgboost.py

Outputs (in NFT_program_V1/results/):
    xgb_model_disc21.pkl
    tuning_results.csv
    tuning_report.md
    training_report.md
    test_report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
V1_DIR       = SCRIPT_DIR.parent
FEATURES_DIR = V1_DIR / "data" / "features"
RESULTS_DIR  = V1_DIR / "results"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
LABEL_COLUMN = "label"

FEATURE_COLUMNS = [
    "dist_aHash", "dist_pHash", "dist_hsvHash", "dist_sHash",
    "rank_aHash", "rank_pHash", "rank_hsvHash", "rank_sHash",
    "n_trees_found", "best_rank",
]

# Base XGBoost parameters (fixed)
XGBOOST_BASE_PARAMS = {
    "eval_metric":      "aucpr",
    "use_label_encoder": False,
    "random_state":     RANDOM_SEED,
}

# GridSearchCV parameter grid (5 × 4 × 4 = 80 combos × 5 folds = 400 fits)
TUNE_GRID = {
    "max_depth":     [3, 4, 5, 6, 8],
    "n_estimators":  [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}


# ---------------------------------------------------------------------------
# Tuning  →  Training  →  Test evaluation
# ---------------------------------------------------------------------------

def run_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load training features ----
    train_path = FEATURES_DIR / "train_features.csv"
    test_path  = FEATURES_DIR / "test_features.csv"

    print("=" * 70)
    print("STEP 6 — XGBoost Training with GridSearchCV (DISC-21, V2 features)")
    print("=" * 70)

    print(f"\n[1] Loading training features from {train_path} ...")
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run step5_generate_features.py first.")
        sys.exit(1)
    train_df = pd.read_csv(train_path)

    missing = [c for c in FEATURE_COLUMNS if c not in train_df.columns]
    if missing:
        print(f"ERROR: Missing feature columns: {missing}")
        sys.exit(1)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[LABEL_COLUMN].values

    n_pos   = int(y_train.sum())
    n_neg   = len(y_train) - n_pos
    ratio   = n_neg / max(n_pos, 1)

    print(f"  Training samples : {len(y_train)}  ({n_pos} positive, {n_neg} negative)")
    print(f"  Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  Imbalance ratio  : 1:{ratio:.1f}  -> scale_pos_weight={ratio:.2f}")

    # ---- GridSearchCV ----
    n_combos = 1
    for v in TUNE_GRID.values():
        n_combos *= len(v)
    print(f"\n[2] GridSearchCV: {n_combos} combos × 5 folds = {n_combos*5} fits ...")
    print(f"  Grid: {TUNE_GRID}")

    base_params = dict(XGBOOST_BASE_PARAMS)
    base_params["scale_pos_weight"] = ratio

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=XGBClassifier(**base_params),
        param_grid=TUNE_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=False,
        return_train_score=False,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    tune_time = time.time() - t0
    print(f"\n  GridSearchCV completed in {tune_time:.1f}s")

    # Display results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    display_cols = [
        "rank_test_score", "mean_test_score", "std_test_score",
        "param_max_depth", "param_n_estimators", "param_learning_rate",
        "mean_fit_time",
    ]
    results_display = results_df[display_cols].copy()
    results_display.columns = [
        "Rank", "Mean_F1", "Std_F1",
        "max_depth", "n_estimators", "learning_rate",
        "Mean_Fit_Time_s",
    ]

    print(f"\n  Top 10 parameter combinations:")
    print(results_display.head(10).to_string(index=False))

    csv_path = RESULTS_DIR / "tuning_results.csv"
    results_display.to_csv(csv_path, index=False)
    print(f"\n  -> Tuning results saved to {csv_path}")

    best_params = grid_search.best_params_
    best_score  = grid_search.best_score_
    print(f"\n  *** Best parameters: {best_params}")
    print(f"  *** Best mean F1   : {best_score:.4f}")

    # Save tuning report (markdown)
    tuning_report_lines = [
        "# Hyperparameter Tuning Report (DISC-21, V2)\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Search time**: {tune_time:.1f}s\n",
        "\n## Best Parameters\n",
        f"- **max_depth**: {best_params['max_depth']}\n",
        f"- **n_estimators**: {best_params['n_estimators']}\n",
        f"- **learning_rate**: {best_params['learning_rate']}\n",
        f"- **Best mean F1**: {best_score:.4f}\n",
        "\n## Search Grid\n",
        f"```json\n{json.dumps(TUNE_GRID, indent=2)}\n```\n",
        "\n## Top 10 Combinations\n",
        "| Rank | max_depth | n_estimators | learning_rate | Mean F1 | Std |\n",
        "|------|-----------|-------------|---------------|---------|-----|\n",
    ]
    for _, row in results_display.head(10).iterrows():
        tuning_report_lines.append(
            f"| {int(row['Rank'])} | {int(row['max_depth'])} | "
            f"{int(row['n_estimators'])} | {row['learning_rate']} "
            f"| {row['Mean_F1']:.4f} | {row['Std_F1']:.4f} |\n"
        )
    tuning_report_path = RESULTS_DIR / "tuning_report.md"
    with open(tuning_report_path, "w", encoding="utf-8") as f:
        f.writelines(tuning_report_lines)
    print(f"  -> Tuning report saved to {tuning_report_path}")

    # ---- Retrain final model with best params ----
    print(f"\n[3] Retraining final model with best parameters ...")

    final_params = dict(XGBOOST_BASE_PARAMS)
    final_params.update(best_params)
    final_params["scale_pos_weight"] = ratio

    print("  XGBoost parameters:")
    for k, v in final_params.items():
        print(f"    {k}: {v}")

    model = XGBClassifier(**final_params)

    # 5-fold CV (to report stable generalisation estimate)
    print("\n  Running 5-fold stratified CV (tuned params) ...")
    t0 = time.time()
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
    ap_scores  = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
    cv_time = time.time() - t0
    print(f"  ROC-AUC per fold : {[f'{s:.4f}' for s in auc_scores]}")
    print(f"  Mean ROC-AUC     : {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
    print(f"  PR-AUC per fold  : {[f'{s:.4f}' for s in ap_scores]}")
    print(f"  Mean PR-AUC      : {ap_scores.mean():.4f} +/- {ap_scores.std():.4f}")
    print(f"  CV done in {cv_time:.1f}s")

    # Fit on full training set
    print("\n  Fitting on full training set ...")
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # Train-set metrics (sanity check)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = model.predict(X_train)
    train_auc = roc_auc_score(y_train, y_prob_train)
    train_ap  = average_precision_score(y_train, y_prob_train)
    print(f"\n  Train ROC-AUC : {train_auc:.4f}")
    print(f"  Train PR-AUC  : {train_ap:.4f}")
    print("\n  Classification report (train):")
    print(classification_report(y_train, y_pred_train,
                                target_names=["Non-duplicate", "Duplicate"]))

    # Feature importance
    importances = model.feature_importances_
    print("  Feature importance:")
    for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        print(f"    {col}: {imp:.4f}")

    # Save model
    model_data = {
        "model":           model,
        "feature_columns": FEATURE_COLUMNS,
        "params":          final_params,
        "scale_pos_weight": ratio,
        "cv_roc_auc_mean": float(auc_scores.mean()),
        "cv_pr_auc_mean":  float(ap_scores.mean()),
        "best_grid_params": best_params,
        "best_grid_f1":    best_score,
    }
    model_path = RESULTS_DIR / "xgb_model_disc21.pkl"
    joblib.dump(model_data, model_path)
    print(f"\n  -> Model saved to {model_path}")

    # Save training report
    cm_train = confusion_matrix(y_train, y_pred_train)
    training_report_lines = [
        "# XGBoost Training Report (DISC-21, V2)\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Dataset\n",
        f"- Training samples: {len(y_train)} ({n_pos} positive, {n_neg} negative)\n",
        f"- Imbalance ratio: 1:{ratio:.0f}\n",
        f"- Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}\n",
        "\n## Cross-Validation (5-fold, tuned params)\n",
        f"- Mean ROC-AUC : {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}\n",
        f"- Mean PR-AUC  : {ap_scores.mean():.4f} +/- {ap_scores.std():.4f}\n",
        "\n## Final Model (full train)\n",
        f"- Train ROC-AUC: {train_auc:.4f}\n",
        f"- Train PR-AUC : {train_ap:.4f}\n",
        f"- TN={cm_train[0,0]}  FP={cm_train[0,1]}  FN={cm_train[1,0]}  TP={cm_train[1,1]}\n",
        "\n## Feature Importance\n",
    ]
    for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        training_report_lines.append(f"- {col}: {imp:.4f}\n")
    training_report_lines.append(
        f"\n## Parameters\n```json\n{json.dumps(final_params, indent=2)}\n```\n"
    )
    with open(RESULTS_DIR / "training_report.md", "w", encoding="utf-8") as f:
        f.writelines(training_report_lines)
    print(f"  -> Training report saved to {RESULTS_DIR / 'training_report.md'}")

    # ---- Test evaluation ----
    print(f"\n[4] Evaluating on held-out test set (IMAGE-level) ...")
    if not test_path.exists():
        print(f"  WARNING: {test_path} not found — skipping test evaluation.")
        return

    test_df = pd.read_csv(test_path)
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[LABEL_COLUMN].values

    n_pos_test = int(y_test.sum())
    n_neg_test = len(y_test) - n_pos_test
    print(f"  Test rows: {len(y_test)} ({n_pos_test} positive pairs, {n_neg_test} negative pairs)")

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)

    test_auc = roc_auc_score(y_test, y_prob_test)
    test_ap  = average_precision_score(y_test, y_prob_test)

    # ------------------------------------------------------------------
    # Image-level evaluation
    #
    # TP = duplicate query whose true-reference pair was predicted positive
    # FN = duplicate query whose true-reference pair was predicted negative
    # FP = negative IMAGE where AT LEAST ONE candidate was predicted positive
    # TN = negative IMAGE where NO candidate was predicted positive
    # ------------------------------------------------------------------
    import csv as _csv

    def _load_fns(csv_path):
        with open(csv_path, encoding="utf-8", newline="") as fh:
            return {r["filename"].strip() for r in _csv.DictReader(fh)}

    SPLITS_DIR   = V1_DIR / "data" / "splits"
    test_dup_fns = _load_fns(SPLITS_DIR / "test_duplications.csv")
    test_neg_fns = _load_fns(SPLITS_DIR / "test_negatives.csv")

    test_df = test_df.copy()
    test_df["y_pred"] = y_pred_test

    # ---- Recall (image-level) ----
    # One row per duplicate query has label=1 (the true reference pair).
    dup_df         = test_df[test_df["query_image"].isin(test_dup_fns)]
    true_ref_pairs = dup_df[dup_df[LABEL_COLUMN] == 1]
    TP_img = int((true_ref_pairs["y_pred"] == 1).sum())
    FN_img = int((true_ref_pairs["y_pred"] == 0).sum())

    # ---- FPR (image-level) ----
    # A negative image is a FP if ANY of its candidates was predicted positive.
    neg_df      = test_df[test_df["query_image"].isin(test_neg_fns)]
    neg_flagged = neg_df.groupby("query_image")["y_pred"].max()  # 1 if any flagged
    FP_img = int((neg_flagged == 1).sum())
    TN_img = int((neg_flagged == 0).sum())

    n_dup_images = len(test_dup_fns)
    n_neg_images = len(test_neg_fns)

    recall_img    = TP_img / (TP_img + FN_img) if (TP_img + FN_img) > 0 else 0.0
    fpr_img       = FP_img / (FP_img + TN_img) if (FP_img + TN_img) > 0 else 0.0
    precision_img = TP_img / (TP_img + FP_img) if (TP_img + FP_img) > 0 else 0.0
    f1_img        = (2 * precision_img * recall_img / (precision_img + recall_img)
                     if (precision_img + recall_img) > 0 else 0.0)

    print(f"\n  ============================================================")
    print(f"  TEST RESULTS — Image-level evaluation")
    print(f"  ============================================================")
    print(f"  Unit = one IMAGE (not one candidate pair)")
    print(f"  Duplicate images : {n_dup_images}  |  Negative images: {n_neg_images}")
    print(f"")
    print(f"  Recall    : {recall_img:.4f}  ({TP_img}/{n_dup_images})")
    print(f"             TP = true reference pair was predicted positive")
    print(f"  FPR       : {fpr_img:.4f}  ({FP_img}/{n_neg_images})")
    print(f"             FP = negative image with >=1 candidate flagged")
    print(f"  Precision : {precision_img:.4f}")
    print(f"  F1        : {f1_img:.4f}")
    print(f"")
    print(f"  TP={TP_img}  FP={FP_img}  FN={FN_img}  TN={TN_img}")
    print(f"  ROC-AUC (pair-level) : {test_auc:.4f}")
    print(f"  PR-AUC  (pair-level) : {test_ap:.4f}")
    print(f"  ============================================================")

    # Save test report
    test_report_lines = [
        "# XGBoost Test Report (DISC-21, V2) — Image-level\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Evaluation Logic\n",
        "- **Unit**: one IMAGE (not one candidate pair)\n",
        "- **TP**: duplicate query whose true-reference pair was predicted positive\n",
        "- **FP**: negative image where AT LEAST ONE top-K candidate was predicted positive\n",
        "- **TN**: negative image where NO candidate was predicted positive\n",
        "- **FN**: duplicate query whose true-reference pair was predicted negative\n",
        f"\n## Image Counts\n",
        f"- Duplicate test images : {n_dup_images}\n",
        f"- Negative test images  : {n_neg_images}\n",
        "\n## Results\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Recall    | {recall_img:.4f} ({TP_img}/{n_dup_images}) |\n",
        f"| FPR       | {fpr_img:.4f} ({FP_img}/{n_neg_images}) |\n",
        f"| Precision | {precision_img:.4f} |\n",
        f"| F1        | {f1_img:.4f} |\n",
        f"| ROC-AUC (pair) | {test_auc:.4f} |\n",
        f"| PR-AUC  (pair) | {test_ap:.4f} |\n",
        f"\n## Confusion Matrix (image-level)\n",
        f"- TP={TP_img}  FP={FP_img}  FN={FN_img}  TN={TN_img}\n",
        "\n## Best Hyperparameters (from GridSearchCV)\n",
        f"- max_depth: {best_params['max_depth']}\n",
        f"- n_estimators: {best_params['n_estimators']}\n",
        f"- learning_rate: {best_params['learning_rate']}\n",
        f"- Best CV F1: {best_score:.4f}\n",
    ]
    test_report_path = RESULTS_DIR / "test_report_xgb.md"
    with open(test_report_path, "w", encoding="utf-8") as f:
        f.writelines(test_report_lines)
    print(f"\n  -> Test report saved to {test_report_path}")

    print(f"\n{'='*70}")
    print(f"  Pipeline complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_pipeline()

