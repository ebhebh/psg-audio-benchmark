"""Anonymous QC figures for Stage 4 (prompt section 4/5).

At most five anonymous plots: core coverage, awake exclusions, event-start
destiny, positive/negative/excluded window totals, and the per-patient usable
window-count distribution. No patient content, no absolute paths in
titles/labels, and NO audio-event mapping plot (audio is unresolved). All data
are passed in as plain numeric structures so the renderer is decoupled from the
runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_window_qc_figures(
    out_dir: Path,
    *,
    run_id: str,
    per_patient_core: Sequence[Tuple[int, int]],          # (candidate, core_complete)
    per_patient_awake_excluded: Sequence[int],
    event_destiny_counts: Dict[str, int],
    window_label_counts: Dict[str, int],                  # positive/negative/excluded
    per_patient_usable: Sequence[int],
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

    # 1) per-patient core coverage: candidate vs core-complete window counts
    if per_patient_core:
        cand = [p[0] for p in per_patient_core]
        comp = [p[1] for p in per_patient_core]
        idx = list(range(1, len(cand) + 1))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(idx, cand, color="#9EC0E4", label="candidate windows")
        ax.bar(idx, comp, color="#4C78A8", label="core-complete windows")
        ax.set_xlabel("anonymous patient rank")
        ax.set_ylabel("window count")
        ax.set_title("Core-signal coverage per patient (anonymous)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save("fig1_core_coverage.png")

    # 2) per-patient awake-excluded window counts
    if per_patient_awake_excluded:
        idx = list(range(1, len(per_patient_awake_excluded) + 1))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(idx, list(per_patient_awake_excluded), color="#E45756")
        ax.set_xlabel("anonymous patient rank")
        ax.set_ylabel("awake-excluded windows")
        ax.set_title("Windows excluded for awake overlap (anonymous)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig2_awake_excluded.png")

    # 3) event-start destiny distribution
    if any(event_destiny_counts.values()):
        labels = sorted(event_destiny_counts)
        vals = [event_destiny_counts[k] for k in labels]
        colors = {"linked": "#54A24B", "awake_overlap_event": "#E45756",
                  "event_start_outside_core_domain": "#9D7660",
                  "event_in_excluded_or_absent_window": "#B279A2"}
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(labels, vals, color=[colors.get(k, "#4C78A8") for k in labels])
        ax.set_ylabel("event count")
        ax.set_title("Event-start destiny (anonymous, all patients)")
        ax.tick_params(axis="x", labelrotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save("fig3_event_destiny.png")

    # 4) positive / negative / excluded window totals
    if any(window_label_counts.values()):
        order = ["positive", "negative", "excluded"]
        labels = [k for k in order if k in window_label_counts]
        vals = [window_label_counts.get(k, 0) for k in labels]
        colors = {"positive": "#54A24B", "negative": "#4C78A8", "excluded": "#E45756"}
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(labels, vals, color=[colors.get(k, "#4C78A8") for k in labels])
        ax.set_ylabel("window count")
        ax.set_title("Research label distribution (anonymous)")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        _save("fig4_label_distribution.png")

    # 5) per-patient usable (positive+negative) window-count distribution
    if per_patient_usable:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(list(per_patient_usable), bins=30, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("usable windows per patient (positive + negative)")
        ax.set_ylabel("patient count")
        ax.set_title("Usable window count per patient (anonymous)")
        ax.grid(axis="y", alpha=0.3)
        _save("fig5_usable_per_patient.png")

    return written


__all__ = ["render_window_qc_figures"]
