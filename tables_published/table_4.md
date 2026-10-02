**Table 4. Airflow secondary paired analysis on the airflow sub-cohort.**
| Quantity | Value (95% CI) |
|---|---|
| Core (HR + SpO₂) AUROC | 0.815 |
| Enhanced (HR + SpO₂ + airflow) AUROC | 0.812 |
| Core AUPRC | 0.699 |
| Enhanced AUPRC | 0.690 |
| ΔAUROC (enhanced − core) | −0.002 (−0.025 to 0.018) |
| ΔAUPRC (enhanced − core) | −0.009 (−0.033 to 0.015) |
| ΔBrier (enhanced − core) | −0.004 (−0.013 to 0.009) |

*Note.* Secondary paired analysis on the 34-patient airflow sub-cohort (21,441 analysable windows, main label, lag-0 features; same patients, same label, same frozen split for both feature sets). Both AUROCs are window-level out-of-fold estimates under the same patient-level nested cross-validation; Δ values are enhanced − core with 95% CIs from a patient-cluster *paired* bootstrap (1,000 resamples; the same patient draw feeds both feature sets). The 95% CI of ΔAUROC includes 0: under this cohort, feature representation and validation design, no clear incremental improvement was observed — the interval does not establish absence of independent airflow information. The displayed absolute AUROCs (0.815, 0.812) differ by −0.003 after rounding to three decimals, whereas the exact frozen ΔAUROC is −0.002; the delta is computed from unrounded source values and must not be recomputed from the displayed rounded numbers. Because the expert reference scoring shares PSG source channels (oximetry, airflow) with the candidate features, the increment is construct-limited (reference-standard feature association); the shared source is a potential explanation for the null increment, not a demonstrated mechanism. The result is exploratory and cohort-restricted. AUPRC = average precision. AUROC = area under the receiver operating characteristic curve; AUPRC = area under the precision–recall curve; CI = confidence interval; Δ = difference (enhanced − core); HR = heart rate; SpO₂ = peripheral oxygen saturation.
