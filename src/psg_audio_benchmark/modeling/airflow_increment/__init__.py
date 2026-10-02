"""Stage-8 airflow-incremental paired nested-CV comparison.

Implements the pre-registered, leak-proof WITHIN-cohort airflow feature-increment
comparison on the frozen approved v2 patient split, restricted to the
34-patient airflow-eligible sub-cohort. Two membership-identical feature sets
(core_restricted = 16 HR+SpO2; airflow_enhanced = those 16 + 11 airflow) are
compared under the SAME inherited patient-level nested CV. Audio is BLOCKED;
airflow enters ONLY the enhanced matrix. Modules:

* :mod:`airflow_increment.schema`          - constants, feature sets, model roles.
* :mod:`airflow_increment.config`          - resolved config loader + writer.
* :mod:`airflow_increment.data_gate`       - v2 gate + airflow sub-cohort gate.
* :mod:`airflow_increment.feature_assembly`- paired frames (identical membership).
* :mod:`airflow_increment.paired_bootstrap`- paired patient-cluster bootstrap.
* :mod:`airflow_increment.runner`          - orchestrator + artifact writing.
* :mod:`airflow_increment.report`          - markdown + figures.
"""

from __future__ import annotations

from .config import (
    AirflowConfigError,
    ResolvedAirflowConfig,
    load_airflow_config,
    write_resolved_airflow_yaml,
)
from .data_gate import AirflowGateError, AirflowGateResult, verify_airflow_gate
from .feature_assembly import (
    AirflowFeatureAssemblyError,
    AirflowModelFrame,
    assemble_paired_frames,
    assert_airflow_set_isolation,
    assert_frames_identical_membership,
)
from .paired_bootstrap import PairedBootstrapCI, paired_patient_cluster_bootstrap
from .runner import (
    AirflowModelingContaminationError,
    Stage8Options,
    Stage8Runner,
    Stage8Summary,
    assert_not_contaminating_airflow,
)

__all__ = [
    "AirflowConfigError",
    "ResolvedAirflowConfig",
    "load_airflow_config",
    "write_resolved_airflow_yaml",
    "AirflowGateError",
    "AirflowGateResult",
    "verify_airflow_gate",
    "AirflowFeatureAssemblyError",
    "AirflowModelFrame",
    "assemble_paired_frames",
    "assert_airflow_set_isolation",
    "assert_frames_identical_membership",
    "PairedBootstrapCI",
    "paired_patient_cluster_bootstrap",
    "AirflowModelingContaminationError",
    "Stage8Options",
    "Stage8Runner",
    "Stage8Summary",
    "assert_not_contaminating_airflow",
]
