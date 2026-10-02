"""Stage-7 core-CSV (HR + SpO2) traditional-model nested-CV baseline.

Implements the pre-registered, leak-proof window-level dataset-scored
respiratory-event baseline on the frozen approved v2 patient split. Audio is
BLOCKED. Modules:

* :mod:`modeling.schema`     - constants, feature allow-list, model roles.
* :mod:`modeling.config`     - resolved config loader + writer.
* :mod:`modeling.data_gate`  - frozen v2 signature/cohort verification gate.
* :mod:`modeling.feature_assembly` - HR/SpO2 join -> ModelFrame (allow-list).
* :mod:`modeling.folds`      - fold CV adapter (frozen assignments; no re-split).
* :mod:`modeling.pipelines`  - 3 estimator pipelines + grids.
* :mod:`modeling.selection`  - inner-CV candidate search.
* :mod:`modeling.threshold`  - inner-OOF Youden threshold.
* :mod:`modeling.metrics`    - metric bundle (NA on single class).
* :mod:`modeling.bootstrap`  - patient-cluster bootstrap CI.
* :mod:`modeling.runner`     - orchestrator + artifact writing.
* :mod:`modeling.report`     - markdown + figures.
"""

from __future__ import annotations

from .config import (
    ModelingConfigError,
    ResolvedModelingConfig,
    load_modeling_config,
    write_resolved_modeling_yaml,
)
from .data_gate import DataGateError, DataGateResult, verify_gate
from .feature_assembly import FeatureAssemblyError, ModelFrame, assemble_model_frame
from .folds import FoldAdapter, FoldError, build_fold_adapter
from .pipelines import Candidate, ModelSpec, build_model_specs, fit_predict
from .selection import InnerSelectionResult, select_inner
from .threshold import ThresholdResult, youden_threshold
from .metrics import compute_metrics
from .bootstrap import BootstrapCI, patient_cluster_bootstrap
from .runner import (
    ModelingContaminationError,
    Stage7Options,
    Stage7Runner,
    Stage7Summary,
    assert_not_contaminating,
)

__all__ = [
    "ModelingConfigError",
    "ResolvedModelingConfig",
    "load_modeling_config",
    "write_resolved_modeling_yaml",
    "DataGateError",
    "DataGateResult",
    "verify_gate",
    "FeatureAssemblyError",
    "ModelFrame",
    "assemble_model_frame",
    "FoldAdapter",
    "FoldError",
    "build_fold_adapter",
    "Candidate",
    "ModelSpec",
    "build_model_specs",
    "fit_predict",
    "InnerSelectionResult",
    "select_inner",
    "ThresholdResult",
    "youden_threshold",
    "compute_metrics",
    "BootstrapCI",
    "patient_cluster_bootstrap",
    "ModelingContaminationError",
    "Stage7Options",
    "Stage7Runner",
    "Stage7Summary",
    "assert_not_contaminating",
]
