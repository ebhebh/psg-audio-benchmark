"""Stage 9 -- results robustness & sensitivity analysis (core / airflow CSV).

Reproduces and stress-tests the FROZEN Stage 7 (core HR+SpO2) and Stage 8 (paired
airflow increment) results against pre-registered label / threshold / calibration
/ class-weight / patient-weight choices, and transparently describes the
selective 34-patient airflow sub-cohort. The Stage 7/8 primary results are
read-only; all outputs are tagged ``analysis_role`` and written to a new isolated
run-dir. Audio is BLOCKED end-to-end.

Modules:
* :mod:`robustness.schema`     - constants, analysis roles, column contracts.
* :mod:`robustness.config`     - resolved config loader + writer.
* :mod:`robustness.signature`  - frozen-input verification + input signature.
* :mod:`robustness.reproduction` - 2.1 primary reproduction from frozen OOF.
* :mod:`robustness.rerun`      - shared frozen nested-CV re-run engine.
* :mod:`robustness.threshold_sensitivity` - 2.3.
* :mod:`robustness.calibration` - 2.4.
* :mod:`robustness.class_weight` - 2.5.
* :mod:`robustness.bootstrap_sensitivity` - 2.2.
* :mod:`robustness.airflow_profile` - 2.6.
* :mod:`robustness.alternative_label` - 3.0.
* :mod:`robustness.report`     - figures + markdown.
* :mod:`robustness.runner`     - orchestrator.
"""

from __future__ import annotations

from .config import (
    RobustnessConfigError,
    ResolvedRobustnessConfig,
    load_robustness_config,
    write_sensitivity_resolved_yaml,
)
from .runner import (
    RobustnessContaminationError,
    Stage9Options,
    Stage9Runner,
    Stage9Summary,
    assert_not_contaminating_robustness,
)
from .signature import RobustnessInputError

__all__ = [
    "RobustnessConfigError",
    "ResolvedRobustnessConfig",
    "load_robustness_config",
    "write_sensitivity_resolved_yaml",
    "RobustnessContaminationError",
    "RobustnessInputError",
    "Stage9Options",
    "Stage9Runner",
    "Stage9Summary",
    "assert_not_contaminating_robustness",
]
