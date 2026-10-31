# XGBoost Training Report (V2)
**Date**: 2026-10-20 14:07:25

## Dataset
- Training samples: 156641 (561 positive, 156080 negative)
- Imbalance ratio: 1:278
- Features (10): ['dist_aHash', 'dist_pHash', 'dist_sHash', 'dist_hsvHash', 'rank_aHash', 'rank_pHash', 'rank_sHash', 'rank_hsvHash', 'n_trees_found', 'best_rank']

## Cross-Validation (5-fold stratified, tuned params)
- Mean ROC-AUC: 0.9894 +/- 0.0050
- Mean PR-AUC:  0.8827 +/- 0.0149

## Final Model (full train)
- Train ROC-AUC: 1.0000
- Train PR-AUC:  1.0000

## Feature Importance
- best_rank: 0.8603
- rank_sHash: 0.0269
- dist_hsvHash: 0.0258
- rank_hsvHash: 0.0204
- dist_pHash: 0.0171
- rank_pHash: 0.0153
- rank_aHash: 0.0145
- dist_aHash: 0.0076
- n_trees_found: 0.0075
- dist_sHash: 0.0046

## Parameters
```json
{
  "max_depth": 5,
  "n_estimators": 500,
  "learning_rate": 0.2,
  "eval_metric": "aucpr",
  "use_label_encoder": false,
  "random_state": 42,
  "scale_pos_weight": 278.2174688057041
}
```
