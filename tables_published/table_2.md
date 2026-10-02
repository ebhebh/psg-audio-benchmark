**Table 2. Window-level discrimination by feature set and learner (main label, lag-0 features).**
| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier |
|---|---|---|---:|
| SpO₂ min (simple baseline, LR) | 0.763 | 0.639 | 0.195 |
| HR only (LR) | 0.668 | 0.550 | 0.223 |
| SpO₂ only (LR) | 0.836 | 0.709 | 0.180 |
| HR + SpO₂ (LR, primary) | 0.823 [0.787, 0.854] | 0.703 [0.616, 0.782] | 0.177 |
| HR + SpO₂ (HGB, exploratory) | 0.855 | 0.751 | 0.153 |

*Note.* Window-level out-of-fold discrimination estimated by patient-level nested cross-validation (5 outer × 4 inner folds; scaling and imputation are fitted only within training partitions — within each inner-training subset during inner selection and, for the final refit, within the full outer-training set — and never on inner-validation or outer-test data), pooled over all 30,506 analysable windows (50 patients; main label; lag-0, zero-future-window features). The pre-designated primary model is HR + SpO₂ with LR; HGB is an exploratory non-linear comparator and is not promoted to primary despite its higher point AUROC. 95% CIs (patient-cluster bootstrap, 1,000 resamples) are tabulated for the primary model only; other rows are point estimates and no CI is implied. SpO₂-only and HR + SpO₂ have similar numerical performance, but no paired CI was computed for this comparison and statistical equivalence is not claimed. AUPRC = average precision. AUROC = area under the receiver operating characteristic curve; AUPRC = area under the precision–recall curve; CI = confidence interval; LR = L2-regularised logistic regression; HGB = histogram gradient-boosting classifier; HR = heart rate; SpO₂ = peripheral oxygen saturation.
