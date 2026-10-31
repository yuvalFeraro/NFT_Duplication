# XGBoost Training Report (DISC-21, V2)
**Date**: 2026-10-28 11:51:02

## Dataset
- Training samples: 133560 (363 positive, 133197 negative)
- Imbalance ratio: 1:367
- Features (10): ['dist_aHash', 'dist_pHash', 'dist_hsvHash', 'dist_sHash', 'rank_aHash', 'rank_pHash', 'rank_hsvHash', 'rank_sHash', 'n_trees_found', 'best_rank']

## Cross-Validation (5-fold, tuned params)
- Mean ROC-AUC : 0.8482 +/- 0.0256
- Mean PR-AUC  : 0.5244 +/- 0.0505

## Final Model (full train)
- Train ROC-AUC: 1.0000
- Train PR-AUC : 1.0000
- TN=133189  FP=8  FN=0  TP=363

## Feature Importance
- best_rank: 0.4104
- rank_aHash: 0.0869
- rank_hsvHash: 0.0868
- n_trees_found: 0.0816
- rank_sHash: 0.0729
- dist_aHash: 0.0567
- dist_pHash: 0.0550
- rank_pHash: 0.0505
- dist_sHash: 0.0500
- dist_hsvHash: 0.0491

## Parameters
```json
{
  "eval_metric": "aucpr",
  "use_label_encoder": false,
  "random_state": 42,
  "learning_rate": 0.1,
  "max_depth": 8,
  "n_estimators": 500,
  "scale_pos_weight": 366.93388429752065
}
```
