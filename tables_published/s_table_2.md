### Supplementary Table S2. Feature manifest: definitions and units.
| Modality | Feature | Statistic | Definition | Unit |
|---|---|---|---|---|
| HR | `hr_mean` | mean | arithmetic mean of the finite samples in the window | bpm |
| HR | `hr_median` | median | 50th percentile of the finite samples in the window | bpm |
| HR | `hr_std_ddof1` | std_ddof1 | sample standard deviation (ddof = 1) of the finite samples | bpm |
| HR | `hr_min` | min | minimum finite sample value in the window | bpm |
| HR | `hr_max` | max | maximum finite sample value in the window | bpm |
| HR | `hr_range` | range | max − min of the finite samples | bpm |
| HR | `hr_iqr` | iqr | interquartile range (75th − 25th percentile) of the finite samples | bpm |
| HR | `hr_slope_per_second` | slope_per_second | least-squares slope of the sample values over time (per second; requires ≥ 2 finite samples) | bpm·s⁻¹ |
| SpO₂ | `spo2_mean` | mean | arithmetic mean of the finite samples in the window | % saturation |
| SpO₂ | `spo2_median` | median | 50th percentile of the finite samples in the window | % saturation |
| SpO₂ | `spo2_std_ddof1` | std_ddof1 | sample standard deviation (ddof = 1) of the finite samples | percentage points |
| SpO₂ | `spo2_min` | min | minimum finite sample value in the window | % saturation |
| SpO₂ | `spo2_max` | max | maximum finite sample value in the window | % saturation |
| SpO₂ | `spo2_range` | range | max − min of the finite samples | percentage points |
| SpO₂ | `spo2_iqr` | iqr | interquartile range (75th − 25th percentile) of the finite samples | percentage points |
| SpO₂ | `spo2_slope_per_second` | slope_per_second | least-squares slope of the sample values over time (per second; requires ≥ 2 finite samples) | percentage points·s⁻¹ |
| Airflow | `airflow_mean` | mean | arithmetic mean of the finite samples in the window | arbitrary units |
| Airflow | `airflow_median` | median | 50th percentile of the finite samples in the window | arbitrary units |
| Airflow | `airflow_std_ddof1` | std_ddof1 | sample standard deviation (ddof = 1) of the finite samples | arbitrary units |
| Airflow | `airflow_min` | min | minimum finite sample value in the window | arbitrary units |
| Airflow | `airflow_max` | max | maximum finite sample value in the window | arbitrary units |
| Airflow | `airflow_range` | range | max − min of the finite samples | arbitrary units |
| Airflow | `airflow_iqr` | iqr | interquartile range (75th − 25th percentile) of the finite samples | arbitrary units |
| Airflow | `airflow_slope_per_second` | slope_per_second | least-squares slope of the sample values over time (per second; requires ≥ 2 finite samples) | arbitrary units·s⁻¹ |
| Airflow | `airflow_rms` | rms | root mean square of the finite samples (airflow amplitude) | arbitrary units |
| Airflow | `airflow_zero_crossing_count` | zero_crossing_count | count of consecutive finite-sample sign changes (airflow rhythm; requires ≥ 2 finite samples) | count (dimensionless) |
| Airflow | `airflow_zero_crossing_rate` | zero_crossing_rate | zero-crossing count per second (crossings·s⁻¹) | s⁻¹ |
