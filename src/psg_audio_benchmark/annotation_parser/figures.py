"""Anonymous timeline QC figures for Stage 3 (prompt section 5).

At most five anonymous plots: event timing, awake raw-vs-canonical, CSV signal
durations, cross-midnight prevalence, and event/awake overlap. No patient
content, no absolute paths in titles/labels, no assumed audio synchronization
(there is no audio-event plot because audio is unresolved). All data passed in
as plain numeric structures so the renderer is decoupled from parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# matplotlib is an installed dependency; import lazily so the renderer stays
# optional to import (the runner guards the call).
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_timeline_qc_figures(
    out_dir: Path,
    *,
    run_id: str,
    event_start_rels: Sequence[float],
    awake_raw_canonical_pairs: Sequence[Tuple[int, int]],
    signal_durations_by_modality: Dict[str, Sequence[float]],
    n_cross_midnight_patients: int,
    n_total_patients: int,
    awake_overlap_fractions: Sequence[float],
) -> List[str]:
    """Render up to 5 anonymous QC PNGs into ``out_dir``; return filenames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _mpl()
    written: List[str] = []

    def _save(name: str) -> None:
        path = out_dir / name
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        written.append(name)

    # 1) event start (relative to record start) distribution
    if event_start_rels:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(list(event_start_rels), bins=60, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("event start relative to record_start (s)")
        ax.set_ylabel("event count")
        ax.set_title("Scored event timing (anonymous, all patients)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig1_event_start_relative.png")

    # 2) per-patient awake raw vs canonical counts
    if awake_raw_canonical_pairs:
        raws = [p[0] for p in awake_raw_canonical_pairs]
        cans = [p[1] for p in awake_raw_canonical_pairs]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.scatter(raws, cans, s=22, alpha=0.7, color="#4C78A8", edgecolor="white")
        lim = max(max(raws, default=1), max(cans, default=1)) + 1
        ax.plot([0, lim], [0, lim], color="#888", lw=1, ls="--", label="canonical = raw")
        ax.set_xlabel("raw awake interval count (per patient)")
        ax.set_ylabel("canonical awake interval count (per patient)")
        ax.set_title("Awake intervals: raw vs canonical (anonymous)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        _save("fig2_awake_raw_vs_canonical.png")

    # 3) CSV signal relative-duration distribution by modality
    if signal_durations_by_modality:
        labels = sorted(signal_durations_by_modality)
        data = [list(signal_durations_by_modality[m]) for m in labels]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#9EC0E4", color="#4C78A8"))
        ax.set_ylabel("relative duration (s)")
        ax.set_title("CSV signal duration by modality (anonymous)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig3_csv_signal_durations.png")

    # 4) cross-midnight prevalence
    if n_total_patients:
        no_xmid = max(n_total_patients - n_cross_midnight_patients, 0)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar(["crosses midnight", "same day"], [n_cross_midnight_patients, no_xmid],
               color=["#E45756", "#4C78A8"])
        ax.set_ylabel("patient count")
        ax.set_title("Cross-midnight recordings (anonymous)")
        for i, v in enumerate([n_cross_midnight_patients, no_xmid]):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        _save("fig4_cross_midnight.png")

    # 5) event / awake overlap fraction distribution
    if awake_overlap_fractions:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(list(awake_overlap_fractions), bins=30, color="#54A24B", edgecolor="white")
        ax.set_xlabel("awake overlap fraction per event")
        ax.set_ylabel("event count")
        ax.set_title("Event overlap with awake intervals (anonymous)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig5_event_awake_overlap.png")

    return written


__all__ = ["render_timeline_qc_figures"]
