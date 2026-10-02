"""Stage 5: CSV physiological signal feature extraction + quality audit.

Reads raw physiological CSV samples (heart_rate / spo2 / airflow) inside each
Stage-4 positive/negative window and emits small, interpretable, input-side
feature tables plus a per-window quality/availability audit.

Hard rules (mirrored from the Stage 1-4 discipline and the prompt):

* Features come ONLY from raw CSV signal samples + window time bounds.
* Label / annotation / sleep-stage columns are on a hard deny-list and never
  become features (see :data:`.schema.FORBIDDEN_FEATURE_INPUTS`).
* ``audio_features_present`` is hard-coded False; no ``*.wav`` is ever read.
* Low-coverage modalities are NULL + a quality code, never zero-filled /
  interpolated / imputed.
* No model matrix, no normalization parameters, no train/test split.
* ``data/raw`` is read-only (proved by a before/after snapshot); Stage 1-4
  historical products are never rewritten.
"""

from __future__ import annotations

from .config import (
    PhysiologyFeatureConfigError,
    load_physiology_feature_config,
    write_resolved_config_yaml,
)
from .qc import (
    FeatureContaminationError,
    assert_not_contaminating,
    assert_paths_clean,
    relativize,
    snapshot_raw,
)
from .runner import Stage5Options, Stage5Runner
from .schema import (
    AUDIO_FEATURES_PRESENT,
    FEATURE_SET_VERSION,
    FeatureExclusion,
    FeatureRow,
    FeatureSummary,
)

__all__ = [
    "Stage5Options",
    "Stage5Runner",
    "FeatureSummary",
    "FeatureRow",
    "FeatureExclusion",
    "FeatureContaminationError",
    "PhysiologyFeatureConfigError",
    "load_physiology_feature_config",
    "write_resolved_config_yaml",
    "assert_not_contaminating",
    "assert_paths_clean",
    "relativize",
    "snapshot_raw",
    "AUDIO_FEATURES_PRESENT",
    "FEATURE_SET_VERSION",
]
