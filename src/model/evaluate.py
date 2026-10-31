"""
evaluate.py
===========
Step 7 of the XGBoost training plan:

  Evaluate the trained XGBoost model on the held-out test features.
  Reports both:
    - Per-pair metrics (ROC-AUC, PR-AUC, precision, recall)
    - Per-query metrics (the real deployment metric):
        For each query image, take max(probability) across all candidates.
        If max_prob > threshold -> predict "fake".

Usage
-----
    python src/model/evaluate.py                # V1
    python src/model/evaluate.py --version v2   # V2
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.train_config import (
    LABEL_COLUMN,
    get_version_config,
)


def _query_level_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """
    Compute query-level (image-level) metrics.

    For each query image:
      - max_prob = max predicted probability across all its candidates
      - query_label = 1 if the query has ANY label=1 candidate, else 0
      - query_pred = 1 if max_prob > threshold, else 0
    """
    query_groups = df.groupby("query_image").agg(
        max_prob=("prob", "max"),
        has_positive=(LABEL_COLUMN, "max"),
    ).reset_index()

    query_groups["query_pred"] = (query_groups["max_prob"] > threshold).astype(int)
    query_groups["query_label"] = query_groups["has_positive"].astype(int)

    y_true = query_groups["query_label"].values
    y_pred = query_groups["query_pred"].values
    y_prob = query_groups["max_prob"].values

    n_fake = int(y_true.sum())
    n_orig = len(y_true) - n_fake

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)

    # ROC-AUC at query level (if both classes present)
    if len(set(y_true)) > 1:
        q_auc = roc_auc_score(y_true, y_prob)
    else:
        q_auc = float("nan")

    return {
        "threshold": threshold,
        "n_queries": len(y_true),
        "n_fake_queries": n_fake,
        "n_orig_queries": n_orig,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": recall,
        "precision": precision,
        "FPR": fpr,
        "F1": 2 * precision * recall / max(precision + recall, 1e-9),
        "query_ROC_AUC": q_auc,
    }


def run_evaluation(version: str = "v1"):
    """Load test features + trained model -> evaluate and write report."""
    vcfg = get_version_config(version)
    features_dir = vcfg["features_dir"]
    models_dir = vcfg["models_dir"]
    reports_dir = vcfg["reports_dir"]
    feature_columns = vcfg["feature_columns"]
    model_filename = vcfg["model_filename"]

    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = models_dir / model_filename
    print(f"[{version.upper()}] Loading model from {model_path} ...")
    model_data = joblib.load(model_path)
    model = model_data["model"]

    # ------------------------------------------------------------------
    # Load test features
    # ------------------------------------------------------------------
    test_path = features_dir / "test_features.csv"
    print(f"Loading test features from {test_path} ...")
    df = pd.read_csv(test_path)

    X = df[feature_columns].values
    y = df[LABEL_COLUMN].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  Test samples: {len(y)} ({n_pos} positive, {n_neg} negative)")
    print(f"  Features ({len(feature_columns)}): {feature_columns}")

    # ------------------------------------------------------------------
    # Per-pair predictions
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X)[:, 1]
    df["prob"] = y_prob

    pair_auc = roc_auc_score(y, y_prob) if len(set(y)) > 1 else float("nan")
    pair_ap = average_precision_score(y, y_prob) if len(set(y)) > 1 else float("nan")

    print(f"\n  Per-pair ROC-AUC: {pair_auc:.4f}")
    print(f"  Per-pair PR-AUC:  {pair_ap:.4f}")

    # ------------------------------------------------------------------
    # Threshold sweep (query-level)
    # ------------------------------------------------------------------
    thresholds = np.arange(0.30, 0.95, 0.05)
    sweep_results = []

    print(f"\n  {'Thresh':>7} {'Recall':>8} {'Prec':>8} {'FPR':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print(f"  {'-'*68}")

    for t in thresholds:
        m = _query_level_metrics(df, threshold=t)
        sweep_results.append(m)
        print(f"  {t:>7.2f} {m['recall']:>8.3f} {m['precision']:>8.3f} "
              f"{m['FPR']:>8.3f} {m['F1']:>8.3f} "
              f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}")

    # Find best threshold (maximise F1)
    best = max(sweep_results, key=lambda m: m["F1"])
    print(f"\n  Best threshold (by F1): {best['threshold']:.2f}")
    print(f"    Recall={best['recall']:.3f}  Precision={best['precision']:.3f}  "
          f"FPR={best['FPR']:.3f}  F1={best['F1']:.3f}")

    # Find threshold with FPR <= 5%
    low_fpr = [m for m in sweep_results if m["FPR"] <= 0.05]
    if low_fpr:
        best_low_fpr = max(low_fpr, key=lambda m: m["recall"])
        print(f"\n  Best threshold with FPR <= 5%: {best_low_fpr['threshold']:.2f}")
        print(f"    Recall={best_low_fpr['recall']:.3f}  FPR={best_low_fpr['FPR']:.3f}")

    # ------------------------------------------------------------------
    # Write evaluation report
    # ------------------------------------------------------------------
    report_lines = [
        f"# XGBoost Evaluation Report - {version.upper()} (Test Set)\n",
        f"\n**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"\n## Per-Pair Metrics\n",
        f"- Test samples: {len(y)} ({n_pos} positive, {n_neg} negative)\n",
        f"- Features ({len(feature_columns)}): {feature_columns}\n",
        f"- ROC-AUC: {pair_auc:.4f}\n",
        f"- PR-AUC:  {pair_ap:.4f}\n",
        f"\n## Query-Level Threshold Sweep\n",
        f"\n| Threshold | Recall | Precision | FPR | F1 | TP | FP | FN | TN |\n",
        f"|-----------|--------|-----------|-----|----|----|----|----|----|n",
    ]
    for m in sweep_results:
        report_lines.append(
            f"| {m['threshold']:.2f} | {m['recall']:.3f} | {m['precision']:.3f} | "
            f"{m['FPR']:.3f} | {m['F1']:.3f} | {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} |\n"
        )

    report_lines.append(f"\n## Recommended Threshold\n")
    report_lines.append(f"- **Best by F1**: {best['threshold']:.2f} "
                        f"(Recall={best['recall']:.3f}, Precision={best['precision']:.3f}, "
                        f"FPR={best['FPR']:.3f})\n")
    if low_fpr:
        report_lines.append(f"- **Best with FPR <= 5%**: {best_low_fpr['threshold']:.2f} "
                            f"(Recall={best_low_fpr['recall']:.3f}, FPR={best_low_fpr['FPR']:.3f})\n")

    report_path = reports_dir / "eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n  -> Evaluation report saved to {report_path}")

    # ------------------------------------------------------------------
    # Update model with best threshold
    # ------------------------------------------------------------------
    model_data["threshold"] = best["threshold"]
    model_path = models_dir / model_filename
    joblib.dump(model_data, model_path)
    print(f"  -> Updated model threshold to {best['threshold']:.2f} in {model_path}")

    print(f"\n{'='*60}")
    print(f" [{version.upper()}] Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1",
                        help="v1 = 4-feature model, v2 = 10-feature model")
    args = parser.parse_args()
    run_evaluation(version=args.version)
