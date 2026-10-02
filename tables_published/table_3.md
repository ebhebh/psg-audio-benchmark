**Table 3. Label-protocol and feature-timing (latency) sensitivity of the primary model.**
| Analysis | Windows | Positive windows | Prevalence | AUROC |
|---|---:|---:|---:|---:|
| **Label protocol** (feature timing fixed at lag 0) | | | | |
| Onset (start-point) | 34,643 | 9,732 | 0.281 | 0.754 |
| Main (overlap ≥ 10 s) | 30,506 | 11,514 | 0.377 | 0.823 |
| Any overlap | 34,643 | 15,651 | 0.452 | 0.814 |
| Coverage ≥ 50% | 26,508 | 7,516 | 0.284 | 0.825 |
| **Feature timing** (main label) | | | | |
| Lag 0 (contemporaneous) | 30,506 | 11,514 | 0.377 | 0.823 |
| Offline +30 s | 29,752 | 11,189 | 0.376 | 0.844 |
| Offline +60 s | 29,203 | 10,937 | 0.375 | 0.812 |

*Note.* Sensitivity analyses for the primary HR + SpO₂ LR model; all values are window-level out-of-fold AUROC from the frozen analysis products. The label protocols define different tasks over partly different window sets (onset and any-overlap label the full 34,643-window candidate grid; the main and coverage ≥ 50% protocols exclude their indeterminate windows, 4,137 and 8,135 respectively), so the AUROCs are not interchangeable estimates of one quantity. The feature-timing branches hold the main label fixed and shift the feature window by +30 s or +60 s; windows whose shifted feature window falls outside the available grid are dropped at the shift step, which excludes 804 and 1,471 windows of the 34,643-window candidate cohort (counted before indeterminate-label exclusion; of these, 50 and 168 are indeterminate), so the main-label analysable set shrinks by a net 754 and 1,303 windows relative to lag 0. Lag 0 uses contemporaneous, zero-future-window features (complete only at the label window's end); the +30 s / +60 s branches use future signal and are offline analyses, not real-time/online estimates. Positive-window counts for the label rows are read from the frozen per-protocol label counts; for the feature-timing rows they are the positive counts of the retained label windows (frozen positive rate × retained windows). AUROC = area under the receiver operating characteristic curve; LR = L2-regularised logistic regression; HR = heart rate; SpO₂ = peripheral oxygen saturation.
