# Data directory — intentionally EMPTY (no data is redistributed here)

**LOCAL_CANDIDATE_NOT_PUBLISHED**

This repository contains **no patient data, no audio, and no per-window derived
data**. The `data/` tree exists only because the pipeline reads/writes under it.

## How to obtain the official dataset

| Field | Value |
|---|---|
| Dataset | Tao et al. 2025, "A multimodal dataset for training deep learning models aimed at detecting and analyzing sleep apnea" |
| Dataset paper DOI | `10.1038/s41597-025-05583-8` |
| **Data DOI** | **`10.57760/sciencedb.19070`** |
| Version used in the manuscript | **V5** |
| Dataset license (data page) | CC BY 4.0 (the paper text itself is CC BY-NC-ND 4.0 — that is a different license and does not govern the data) |
| Official data page | https://www.scidb.cn/detail?dataSetId=7b5f1df9c3d4435baa0ff2dbecf487d9&version=V5 |
| Size of the raw tree used | ~53 GB (50 patients; bedside smartphone + digital-recorder audio, SpO2/HR/airflow at 1 Hz, PSG AASM annotations) |

Steps:

1. Register/visit the official Science Data Bank page via the Data DOI above and
   download version **V5** following their terms (CC BY 4.0).
2. Preserve the source tree layout `V5/Data/<patient>/<file>` while downloading
   into `data/external/incoming/` (staging).
3. Run `python scripts/02_download_from_urllist.py --list-only` style audits, or
   place files manually; then run the pipeline stages in order (see README).
4. The pipeline never writes into raw data locations; it audits staged files and
   builds derived features under `features/`.

## Redistribution policy (stricter than the license)

Although the dataset page records CC BY 4.0, this project deliberately
redistributes **nothing** from the dataset: no raw signals, no audio, no
per-window OOF predictions, no patient-to-recording mappings, and no exact
per-patient timelines. If you use the dataset, cite the data DOI and version and
comply with CC BY 4.0 attribution requirements yourself.
