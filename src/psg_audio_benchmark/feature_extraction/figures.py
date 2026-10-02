"""Anonymous QC figures for Stage 5 (prompt section 4/5).

At most five anonymous plots: per-modality coverage distribution, quality-status
counts, per-patient core HR+SpO2 available window counts, per-patient airflow
available window counts, and low-coverage/unavailable counts per modality. No
patient content, no absolute paths, and NO audio figure. All data are passed in
as plain numeric structures so the renderer is decoupled from the runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_physiology_qc_figures(
    out_dir: Path,
    *,
    run_id: str,
    coverage_by_modality: Dict[str, Sequence[float]],   # modality -> coverage_fraction list
    quality_counts_by_modality: Dict[str, Dict[str, int]],
    per_patient_core_available: Sequence[int],
    per_patient_airflow_available: Sequence[int],
    low_or_unavailable_by_modality: Dict[str, int],
) -> List[str]:
    """Render up to 5 anonymous QC PNGs into ``out_dir``; return filenames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _mpl()
    written: List[str] = []
    _C = {"heart_rate": "#4C78A8", "spo2": "#54A24B", "airflow": "#E45756"}

    def _save(name: str) -> None:
        plt.savefig(out_dir / name, dpi=110, bbox_inches="tight")
        plt.close()
        written.append(name)

    # 1) coverage_fraction distribution per modality
    any_cov = any(len(v) for v in coverage_by_modality.values())
    if any_cov:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for mod in ("heart_rate", "spo2", "airflow"):
            cov = [c for c in coverage_by_modality.get(mod, []) if c is not None and np.isfinite(c)]
            if cov:
                ax.hist(cov, bins=20, alpha=0.55, label=mod, color=_C.get(mod, "#4C78A8"))
        ax.axvline(0.80, color="#333333", linestyle="--", linewidth=1, label="min coverage 0.80")
        ax.set_xlabel("coverage fraction (n_finite / n_expected)")
        ax.set_ylabel("window count")
        ax.set_title("Per-modality coverage distribution (anonymous)")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save("fig1_coverage_distribution.png")

    # 2) quality-status counts per modality (grouped bar)
    statuses = [
        "available", "low_coverage", "all_missing",
        "insufficient_samples", "modality_not_available_for_patient",
    ]
    short = {
        "available": "avail",
        "low_coverage": "low_cov",
        "all_missing": "all_miss",
        "insufficient_samples": "few_samp",
        "modality_not_available_for_patient": "no_modality",
    }
    mods = [m for m in ("heart_rate", "spo2", "airflow") if m in quality_counts_by_modality]
    if mods:
        x = np.arange(len(statuses))
        w = 0.8 / max(1, len(mods))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for i, mod in enumerate(mods):
            vals = [int(quality_counts_by_modality.get(mod, {}).get(s, 0)) for s in statuses]
            ax.bar(x + i * w, vals, width=w, label=mod, color=_C.get(mod, "#4C78A8"))
        ax.set_xticks(x + w * (len(mods) - 1) / 2)
        ax.set_xticklabels([short[s] for s in statuses], rotation=15)
        ax.set_ylabel("window count")
        ax.set_title("Quality-status counts per modality (anonymous)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save("fig2_quality_status_counts.png")

    # 3) per-patient core HR+SpO2 available window counts
    if per_patient_core_available:
        idx = list(range(1, len(per_patient_core_available) + 1))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(idx, list(per_patient_core_available), color="#4C78A8")
        ax.set_xlabel("anonymous patient rank")
        ax.set_ylabel("core HR+SpO2 available windows")
        ax.set_title("Core HR+SpO2 available windows per patient (anonymous)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig3_core_available_per_patient.png")

    # 4) per-patient airflow available window counts (only patients with airflow)
    if per_patient_airflow_available:
        idx = list(range(1, len(per_patient_airflow_available) + 1))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(idx, list(per_patient_airflow_available), color="#E45756")
        ax.set_xlabel("anonymous patient rank (airflow patients only)")
        ax.set_ylabel("airflow available windows")
        ax.set_title("Airflow available windows per patient (anonymous)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig4_airflow_available_per_patient.png")

    # 5) low-coverage / unavailable counts per modality
    if any(v for v in low_or_unavailable_by_modality.values()):
        labels = list(low_or_unavailable_by_modality.keys())
        vals = [int(low_or_unavailable_by_modality.get(k, 0)) for k in labels]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(labels, vals, color=[_C.get(k, "#4C78A8") for k in labels])
        ax.set_ylabel("window count (low coverage / unavailable)")
        ax.set_title("Unavailable / low-coverage windows per modality (anonymous)")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        _save("fig5_low_coverage_per_modality.png")

    return written


__all__ = ["render_physiology_qc_figures"]
