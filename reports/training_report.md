# XGBoost Training Report (V1)
**Date**: 2026-10-15 13:06:52

## Dataset
- Training samples: 156648 (561 positive, 156087 negative)
- Imbalance ratio: 1:278
- Features (4): ['dist_aHash', 'dist_pHash', 'dist_sHash', 'dist_hsvHash']

## Cross-Validation (5-fold stratified)
- Mean ROC-AUC: 0.9699 +/- 0.0055
- Mean PR-AUC:  0.7428 +/- 0.0346

## Final Model (full train)
- Train ROC-AUC: 1.0000
- Train PR-AUC:  0.9981

## Feature Importance
- dist_pHash: 0.6762
- dist_hsvHash: 0.1992
- dist_aHash: 0.0664
- dist_sHash: 0.0582

## Parameters
```json
{
  "max_depth": 8,
  "n_estimators": 500,
  "learning_rate": 0.2,
  "eval_metric": "aucpr",
  "use_label_encoder": false,
  "random_state": 42,
  "scale_pos_weight": 278.2299465240642
}
```
