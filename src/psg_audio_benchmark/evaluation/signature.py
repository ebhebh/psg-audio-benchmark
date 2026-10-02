"""Stage-6 input version signature.

Pins the exact Stage 3/4/5 inputs Stage 6 consumes: per-file relative path,
SHA-256, row count and a canonical schema fingerprint, plus each upstream stage's
run id and config hash. A lineage check asserts the Stage-5 resolved config's
``input_stage4_run_id`` equals the Stage-4 run id consumed here (transitive-
leakage guard), and the Stage-4 directory run id agrees with the ``run_id``
column inside the parquet.

``compare_signatures`` rejects a run on run-id / relpath / row-count / schema /
sha256 mismatch (the test-8 gate). Byte-hash equality is version-scoped
(``scikit-learn`` / pandas / pyarrow versions are recorded); within a fixed
environment the gate is exact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..run_metadata import sha256_text

_IGNORED_SUFFIXES = (".png", ".jpg", ".jpeg")


class SignatureMismatchError(RuntimeError):
    """Raised when two input signatures disagree on a content field."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_fingerprint(df: pd.DataFrame) -> str:
    """Canonical sha256 over sorted ``col:numpy_dtype_char`` (stable across versions).

    ``pandas_dtype`` must receive the column's *dtype* (or a dtype string), NOT the
    Series itself -- passing a Series raises ``TypeError: dtype '...' not
    understood`` (the whole Series is stringified), which would silently degrade
    every record to ``unreadable``. ``.char`` is the numpy type code (object->'O',
    float64->'d', bool->'?'); the ``getattr`` fallback covers pandas extension
    dtypes that have no ``.char``.
    """
    pairs: List[str] = []
    for c in df.columns:
        dt = pd.api.types.pandas_dtype(df[c].dtype)
        pairs.append(f"{c}:{getattr(dt, 'char', str(dt))}")
    return sha256_text("\n".join(sorted(pairs)))


def _file_record(path: Path, project_root: Path) -> Dict[str, Any]:
    try:
        rel = str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        rel = path.name
    rec: Dict[str, Any] = {"relpath": rel, "sha256": _sha256_file(path)}
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
            rec["rows"] = int(df.shape[0])
            rec["schema_fingerprint"] = _schema_fingerprint(df)
        except Exception:
            rec["rows"] = -1
            rec["schema_fingerprint"] = "unreadable_parquet"
    elif path.suffix == ".csv":
        try:
            df = pd.read_csv(path)
            rec["rows"] = int(df.shape[0])
            rec["schema_fingerprint"] = _schema_fingerprint(df)
        except Exception:
            rec["rows"] = -1
            rec["schema_fingerprint"] = "unreadable_csv"
    else:
        rec["rows"] = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")))
        rec["schema_fingerprint"] = "text"
    return rec


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def build_input_signature(
    *,
    cfg,
    stage4_run_id: str,
    stage5_run_id: str,
    stage3_run_id: str,
    run_id: str,
    project_root: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    """Build the input version signature dict + a list of lineage anomalies."""
    anomalies: List[str] = []
    s4_dir = cfg.path("data_manifests") / "runs" / stage4_run_id
    s5_dir = cfg.path("features_physiology") / "runs" / stage5_run_id

    inputs: Dict[str, Any] = {}

    def _add_dir(run_dir: Path, prefix: str) -> None:
        if not run_dir.is_dir():
            anomalies.append(f"{prefix}_run_dir_absent:{run_dir.name}")
            return
        for p in sorted(run_dir.iterdir()):
            if not p.is_file() or p.suffix in _IGNORED_SUFFIXES or p.stat().st_size == 0:
                continue
            inputs[f"{prefix}/{p.name}"] = _file_record(p, project_root)

    _add_dir(s4_dir, "stage4")
    _add_dir(s5_dir, "stage5")

    # config hashes + lineage from the resolved yamls
    s4_resolved = _read_yaml(s4_dir / "windowing_config_resolved.yaml")
    s5_resolved = _read_yaml(s5_dir / "physiology_features_resolved.yaml")
    s4_config_hash = str(s4_resolved.get("config_hash", ""))
    s5_config_hash = str(s5_resolved.get("config_hash", ""))
    s5_input_stage4 = str(s5_resolved.get("input_stage4_run_id", ""))

    # lineage: Stage 5 must have been built from THIS Stage 4
    if s5_input_stage4 and s5_input_stage4 != stage4_run_id:
        anomalies.append(
            f"lineage_stage5_input_stage4_mismatch:{s5_input_stage4}!={stage4_run_id}"
        )

    # Stage 4 directory run id must agree with the run_id column inside the parquet
    try:
        win = pd.read_parquet(s4_dir / "csv_window_index.parquet")
        if "run_id" in win.columns and win.shape[0]:
            uniq = set(win["run_id"].dropna().astype(str).unique())
            if uniq != {stage4_run_id}:
                anomalies.append(
                    f"stage4_dir_vs_parquet_run_id_mismatch:dir={stage4_run_id}:parquet={sorted(uniq)}"
                )
    except Exception as exc:  # pragma: no cover - defensive
        anomalies.append(f"stage4_run_id_check_unreadable:{type(exc).__name__}")

    try:
        import sklearn
        sklearn_version = str(getattr(sklearn, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        sklearn_version = "not_installed"

    sig = {
        "run_id": run_id,
        "patient_split_stage": "stage6",
        "stage4_run_id": stage4_run_id,
        "stage5_run_id": stage5_run_id,
        "stage3_run_id": stage3_run_id,
        "stage4_config_hash": s4_config_hash,
        "stage5_config_hash": s5_config_hash,
        "lineage_stage5_input_stage4_run_id": s5_input_stage4,
        "lineage_stage5_matches_stage4": (s5_input_stage4 == stage4_run_id),
        "sklearn_version": sklearn_version,
        "inputs": inputs,
    }
    return sig, anomalies


def compare_signatures(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    """Compare two input signatures. Raise on content mismatch; return byte-hash warnings.

    Raises :class:`SignatureMismatchError` on run-id / relpath / row-count /
    schema-fingerprint disagreement. SHA-256 disagreement is recorded as a
    returned warning (parquet bytes are pyarrow-version dependent) rather than a
    hard error, unless it coincides with a content mismatch.
    """
    mismatches: List[str] = []
    warnings: List[str] = []
    for top in ("stage4_run_id", "stage5_run_id", "stage3_run_id"):
        if expected.get(top) and actual.get(top) and expected[top] != actual[top]:
            mismatches.append(f"{top}:{expected[top]}!={actual[top]}")

    e_in = expected.get("inputs", {})
    a_in = actual.get("inputs", {})
    for key in sorted(set(e_in) | set(a_in)):
        if key not in e_in:
            mismatches.append(f"{key}:missing_in_expected")
            continue
        if key not in a_in:
            mismatches.append(f"{key}:missing_in_actual")
            continue
        e, a = e_in[key], a_in[key]
        if e.get("relpath") != a.get("relpath"):
            mismatches.append(f"{key}:relpath:{e.get('relpath')}!={a.get('relpath')}")
        if e.get("rows") != a.get("rows"):
            mismatches.append(f"{key}:rows:{e.get('rows')}!={a.get('rows')}")
        if e.get("schema_fingerprint") != a.get("schema_fingerprint"):
            mismatches.append(f"{key}:schema:{e.get('schema_fingerprint')}!={a.get('schema_fingerprint')}")
        if e.get("sha256") != a.get("sha256"):
            warnings.append(f"{key}:sha256_drift (byte-level; version-scoped)")

    if mismatches:
        raise SignatureMismatchError(
            "input signature mismatch: " + "; ".join(mismatches)
        )
    return warnings


__all__ = [
    "SignatureMismatchError",
    "build_input_signature",
    "compare_signatures",
]
