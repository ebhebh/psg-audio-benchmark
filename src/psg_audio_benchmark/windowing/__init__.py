"""Stage 4: CSV/annotation-side window index + research label table.

Builds a reproducible, patient-level, half-open, 30 s window grid over each
patient's verified heart_rate ∩ spo2 coverage domain, links standardized
respiratory events by their start point, and assigns mutually-exclusive research
labels (positive / negative / excluded) -- CSV/annotation side only.

This stage does NOT window audio, extract features, map PSG events to WAV,
train, split or evaluate. ``audio_window_eligible`` is hard-coded False for
every window. See ``docs/window_label_contract.md``.
"""

from __future__ import annotations

from .config import WindowingConfigError, load_windowing_config
from .qc import WindowingContaminationError
from .runner import Stage4Options, Stage4Runner
from .schema import WindowingSummary

__all__ = [
    "Stage4Options",
    "Stage4Runner",
    "WindowingSummary",
    "WindowingContaminationError",
    "WindowingConfigError",
    "load_windowing_config",
]
