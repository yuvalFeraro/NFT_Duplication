# XGBoost Test Report (DISC-21, V2)
**Date**: 2026-10-28 11:51:02

## Test Set
- Test samples: 19429 (103 positive, 19326 negative)

## Results
| Metric | Value |
|--------|-------|
| ROC-AUC   | 0.8641 |
| PR-AUC    | 0.5127 |
| Recall    | 0.4757 (49/103) |
| FPR       | 0.0008 (15/19326) |
| Precision | 0.7656 |
| F1        | 0.5868 |

## Confusion Matrix
- TP=49  FP=15  FN=54  TN=19311

## Best Hyperparameters (from GridSearchCV)
- max_depth: 8
- n_estimators: 500
- learning_rate: 0.1
- Best CV F1: 0.5860
