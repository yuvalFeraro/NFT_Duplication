"""
train_xgboost.py
================
Step 6 of the XGBoost training plan:

  Train an XGBoost binary classifier on the distance-vector features
  produced by generate_features.py.

  Always runs GridSearchCV hyperparameter tuning, then retrains the
  final model with the winning parameters.

Usage
-----
    python src/model/train_xgboost.py                # V1 (4 features)
    python src/model/train_xgboost.py --version v2   # V2 (10 features)
"""

import sys
import time
import json
import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.train_config import (
    LABEL_COLUMN,
    XGBOOST_PARAMS,
    XGBOOST_TUNE_GRID,
    RANDOM_SEED,
    get_version_config,
)


def run_training(
    version: str,
    X,
    y,
    ratio: float,
    best_params: dict,
    vcfg: dict,
):
    """
    Fit XGBoost with best_params, evaluate with 5-fold CV, save model.

    Parameters
    ----------
    version     : "v1" or "v2"
    X, y        : pre-loaded feature matrix and label vector (from run_tuning)
    ratio       : neg/pos imbalance ratio (used for scale_pos_weight)
    best_params : winning hyperparameters from GridSearchCV
    vcfg        : version config dict (paths, feature_columns, model_filename, …)
    """
    models_dir = vcfg["models_dir"]
    reports_dir = vcfg["reports_dir"]
    feature_columns = vcfg["feature_columns"]
    model_filename = vcfg["model_filename"]

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    # ------------------------------------------------------------------
    # Configure XGBoost with tuned hyperparameters
    # ------------------------------------------------------------------
    params = dict(XGBOOST_PARAMS)
    params.update(best_params)          # Override with GridSearch winners
    params["scale_pos_weight"] = ratio
    params["random_state"] = RANDOM_SEED

    print(f"\nXGBoost parameters (tuned):")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = XGBClassifier(**params)

    # ------------------------------------------------------------------
    # 5-fold stratified cross-validation (evaluation of tuned params)
    # ------------------------------------------------------------------
    print("\nRunning 5-fold stratified cross-validation ...")
    start = time.time()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # ROC-AUC
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  ROC-AUC per fold: {[f'{s:.4f}' for s in auc_scores]}")
    print(f"  Mean ROC-AUC:     {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")

    # Average Precision
    ap_scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"  PR-AUC per fold:  {[f'{s:.4f}' for s in ap_scores]}")
    print(f"  Mean PR-AUC:      {ap_scores.mean():.4f} +/- {ap_scores.std():.4f}")

    cv_time = time.time() - start
    print(f"  CV completed in {cv_time:.1f}s")

    # ------------------------------------------------------------------
    # Train final model on full training set
    # ------------------------------------------------------------------
    print("\nTraining final model on full training set ...")
    start = time.time()
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    train_time = time.time() - start
    print(f"  Training completed in {train_time:.1f}s")

    # Train set performance (sanity check)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    train_auc = roc_auc_score(y, y_prob)
    train_ap = average_precision_score(y, y_prob)
    print(f"\n  Train ROC-AUC: {train_auc:.4f}")
    print(f"  Train PR-AUC:  {train_ap:.4f}")
    print(f"\n  Classification report (train):")
    print(classification_report(y, y_pred, target_names=["Non-duplicate", "Duplicate"]))

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion matrix (train):")
    print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importances = model.feature_importances_
    print(f"\n  Feature importance:")
    for col, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        print(f"    {col}: {imp:.4f}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_data = {
        "model": model,
        "version": version,
        "feature_columns": feature_columns,
        "threshold": 0.5,
        "cv_roc_auc_mean": float(auc_scores.mean()),
        "cv_pr_auc_mean": float(ap_scores.mean()),
        "scale_pos_weight": ratio,
        "params": params,
        "tuned": True,
    }

    model_path = models_dir / model_filename
    joblib.dump(model_data, model_path)
    print(f"\n  -> Model saved to {model_path}")

    # ------------------------------------------------------------------
    # Save training summary report
    # ------------------------------------------------------------------
    report_lines = [
        f"# XGBoost Training Report ({version.upper()})\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"\n## Dataset\n",
        f"- Training samples: {len(y)} ({n_pos} positive, {n_neg} negative)\n",
        f"- Imbalance ratio: 1:{ratio:.0f}\n",
        f"- Features ({len(feature_columns)}): {feature_columns}\n",
        f"\n## Cross-Validation (5-fold stratified, tuned params)\n",
        f"- Mean ROC-AUC: {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}\n",
        f"- Mean PR-AUC:  {ap_scores.mean():.4f} +/- {ap_scores.std():.4f}\n",
        f"\n## Final Model (full train)\n",
        f"- Train ROC-AUC: {train_auc:.4f}\n",
        f"- Train PR-AUC:  {train_ap:.4f}\n",
        f"\n## Feature Importance\n",
    ]
    for col, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        report_lines.append(f"- {col}: {imp:.4f}\n")

    report_lines.append(f"\n## Parameters\n```json\n{json.dumps(params, indent=2)}\n```\n")

    report_path = reports_dir / "training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"  -> Training report saved to {report_path}")

    print(f"\n{'='*60}")
    print(f" [{version.upper()}] Training pipeline complete!")
    print(f"{'='*60}")

    return model


# ======================================================================
# Hyperparameter tuning via GridSearchCV  →  always the entry point
# ======================================================================

def run_tuning(version: str = "v1"):
    """
    Load features once, run GridSearchCV to find best hyperparameters,
    then pass the data + best params directly to run_training()
    (no second CSV read).
    """
    vcfg = get_version_config(version)
    features_dir = vcfg["features_dir"]
    reports_dir = vcfg["reports_dir"]
    feature_columns = vcfg["feature_columns"]

    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load training features  (single load — reused in run_training)
    # ------------------------------------------------------------------
    train_path = features_dir / "train_features.csv"
    print(f"[{version.upper()}] Loading training features from {train_path} ...")
    df = pd.read_csv(train_path)

    X = df[feature_columns].values
    y = df[LABEL_COLUMN].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ratio = n_neg / max(n_pos, 1)

    print(f"  Samples: {len(y)} ({n_pos} positive, {n_neg} negative)")
    print(f"  Feature columns ({len(feature_columns)}): {feature_columns}")

    # ------------------------------------------------------------------
    # Set up base estimator (fixed params minus the ones being searched)
    # ------------------------------------------------------------------
    base_params = dict(XGBOOST_PARAMS)
    base_params["scale_pos_weight"] = ratio
    base_params["random_state"] = RANDOM_SEED
    # Remove keys that will be searched over so there's no conflict
    for key in XGBOOST_TUNE_GRID:
        base_params.pop(key, None)

    base_model = XGBClassifier(**base_params)

    # ------------------------------------------------------------------
    # GridSearchCV
    # ------------------------------------------------------------------
    n_combos = 1
    for vals in XGBOOST_TUNE_GRID.values():
        n_combos *= len(vals)
    print(f"\n  Grid: {XGBOOST_TUNE_GRID}")
    print(f"  Total combinations: {n_combos}  (× 5 folds = {n_combos * 5} fits)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print("\nRunning GridSearchCV (scoring=f1) ...")
    start = time.time()

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGBOOST_TUNE_GRID,
        scoring="f1",                  # Optimize for best F1 at decision boundary
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=False,                   # Final model trained below via run_training
        return_train_score=False,
    )
    grid_search.fit(X, y)

    tune_time = time.time() - start
    print(f"\n  GridSearchCV completed in {tune_time:.1f}s")

    # ------------------------------------------------------------------
    # Extract and display results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")

    display_cols = [
        "rank_test_score",
        "mean_test_score", "std_test_score",
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

    # Save full results CSV
    csv_path = reports_dir / "tuning_results.csv"
    results_display.to_csv(csv_path, index=False)
    print(f"\n  -> Full tuning results saved to {csv_path}")

    # Best parameters
    best = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"\n  *** Best parameters: {best}")
    print(f"  *** Best mean F1: {best_score:.4f}")

    # ------------------------------------------------------------------
    # Save tuning report (markdown)
    # ------------------------------------------------------------------
    report_lines = [
        f"# Hyperparameter Tuning Report ({version.upper()})\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Search time**: {tune_time:.1f}s\n",
        f"\n## Best Parameters\n",
        f"- **max_depth**: {best['max_depth']}\n",
        f"- **n_estimators**: {best['n_estimators']}\n",
        f"- **learning_rate**: {best['learning_rate']}\n",
        f"- **Best mean F1**: {best_score:.4f}\n",
        f"\n## Search Grid\n",
        f"```json\n{json.dumps(XGBOOST_TUNE_GRID, indent=2)}\n```\n",
        f"\n## Top 10 Combinations\n",
        f"\n| Rank | max_depth | n_estimators | learning_rate | Mean F1 | Std |\n",
        f"|------|-----------|-------------|---------------|---------|-----|\n",
    ]
    for _, row in results_display.head(10).iterrows():
        report_lines.append(
            f"| {int(row['Rank'])} | {int(row['max_depth'])} | {int(row['n_estimators'])} "
            f"| {row['learning_rate']} | {row['Mean_F1']:.4f} | {row['Std_F1']:.4f} |\n"
        )

    report_path = reports_dir / "tuning_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"  -> Tuning report saved to {report_path}")

    # ------------------------------------------------------------------
    # Retrain with best parameters  (data already in memory — no re-read)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Retraining with best parameters ...")
    print(f"{'='*60}\n")

    run_training(
        version=version,
        X=X,
        y=y,
        ratio=ratio,
        best_params=best,
        vcfg=vcfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model (always with GridSearchCV hyperparameter tuning)"
    )
    parser.add_argument(
        "--version", choices=["v1", "v2"], default="v1",
        help="v1 = 4-feature model, v2 = 10-feature model",
    )
    args = parser.parse_args()

    run_tuning(version=args.version)
