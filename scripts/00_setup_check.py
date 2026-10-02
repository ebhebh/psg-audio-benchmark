#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 0 environment check.

Performs REAL detection of the host environment (no guessing) and writes a
structured JSON result plus a human-readable Markdown report. It never
downloads anything, installs packages, upgrades drivers/CUDA, or touches
``data/raw`` beyond an existence/permission check.

Status vocabulary: ``PASS`` | ``WARNING`` | ``FAIL`` | ``NOT_APPLICABLE``.
A missing GPU is at most a WARNING and never blocks the CPU main line.

CLI:
    python scripts/00_setup_check.py [--config CONFIG] [--dry-run] [--run-id ID]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make ``src`` importable when run directly (no install required).
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark import run_metadata  # noqa: E402
from psg_audio_benchmark.config import Config, ConfigError, default_project_root  # noqa: E402
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"
NA = "NOT_APPLICABLE"

#: Packages inspected by this check.
CHECKED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "soundfile",
    "librosa",
    "matplotlib",
    "pyyaml",
    "pytest",
    "torch",
    "lightgbm",
    "xgboost",
)

#: Packages needed for the config/test infra of Stage 0 itself.
STAGE0_REQUIRED = ("pyyaml", "pytest")

#: Packages needed for the CPU main line (feature + baseline ML).
CPU_MAINLINE_REQUIRED = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "soundfile",
    "librosa",
    "matplotlib",
    "lightgbm",
    "xgboost",
)

#: Minimum recommended free disk space (GB).
MIN_FREE_DISK_GB = 20.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], timeout: int = 15) -> Tuple[Optional[int], str, str]:
    """Run a command, never raising. Returns (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return None, "", "command not found"
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, "", f"error: {exc}"


def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _gb(bytes_value: Optional[int]) -> Optional[float]:
    if bytes_value is None:
        return None
    return round(bytes_value / (1024 ** 3), 2)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_os(project_root: Path) -> Dict[str, Any]:
    import platform

    cwd = Path.cwd()
    return {
        "status": PASS,
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "python_platform": sys.platform,
        "cwd": str(cwd),
        "project_root": str(project_root),
        "project_root_non_ascii": _has_non_ascii(str(project_root)),
    }


def check_python() -> Dict[str, Any]:
    in_venv = (
        hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix
    ) or bool(os.environ.get("VIRTUAL_ENV"))
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    return {
        "status": PASS,
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "implementation": platform_implementation(),
        "in_venv": bool(in_venv),
        "conda_env": conda_env,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", None),
    }


def platform_implementation() -> str:
    import platform

    return platform.python_implementation()


def check_hardware() -> Dict[str, Any]:
    import platform

    logical = os.cpu_count()
    physical = None
    mem_total_gb = None
    mem_available_gb = None
    notes: List[str] = []

    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False) or None
        vm = psutil.virtual_memory()
        mem_total_gb = _gb(vm.total)
        mem_available_gb = _gb(vm.available)
    except Exception:
        notes.append("psutil not available; physical cores/memory via fallback")
        physical = _wmic_physical_cores()
        total_b, free_b = _wmic_memory()
        mem_total_gb = _gb(total_b)
        mem_available_gb = _gb(free_b)

    return {
        "status": PASS,
        "processor": platform.processor() or None,
        "logical_cores": logical,
        "physical_cores": physical,
        "memory_total_gb": mem_total_gb,
        "memory_available_gb": mem_available_gb,
        "notes": notes,
    }


def _wmic_physical_cores() -> Optional[int]:
    rc, out, _ = _run(
        ["wmic", "CPU", "get", "NumberOfCores", "/value"], timeout=20
    )
    if rc == 0 and out:
        m = re.search(r"NumberOfCores=(\d+)", out)
        if m:
            return int(m.group(1))
    return None


def _wmic_memory() -> Tuple[Optional[int], Optional[int]]:
    rc, out, _ = _run(
        [
            "wmic",
            "OS",
            "get",
            "TotalVisibleMemorySize,FreePhysicalMemory",
            "/value",
        ],
        timeout=20,
    )
    total_kb = free_kb = None
    if rc == 0 and out:
        tm = re.search(r"TotalVisibleMemorySize=(\d+)", out)
        fm = re.search(r"FreePhysicalMemory=(\d+)", out)
        if tm:
            total_kb = int(tm.group(1))
        if fm:
            free_kb = int(fm.group(1))
    # wmic returns KB; convert to bytes
    total_b = total_kb * 1024 if total_kb is not None else None
    free_b = free_kb * 1024 if free_kb is not None else None

    # Total physical memory fallback (TotalVisibleMemorySize is usable, ~RAM)
    if total_b is None:
        rc2, out2, _ = _run(
            ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
            timeout=20,
        )
        if rc2 == 0 and out2:
            m = re.search(r"TotalPhysicalMemory=(\d+)", out2)
            if m:
                total_b = int(m.group(1))
    return total_b, free_b


def check_disk(project_root: Path) -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(str(project_root))
        free_gb = _gb(usage.free)
        status = WARNING if (free_gb is not None and free_gb < MIN_FREE_DISK_GB) else PASS
        return {
            "status": status,
            "drive": str(project_root.anchor) or None,
            "total_gb": _gb(usage.total),
            "used_gb": _gb(usage.used),
            "free_gb": free_gb,
            "min_recommended_free_gb": MIN_FREE_DISK_GB,
        }
    except Exception as exc:
        return {"status": WARNING, "error": str(exc)}


def check_git(project_root: Path) -> Dict[str, Any]:
    git_bin = shutil.which("git")
    info = run_metadata.git_info(project_root)
    info["git_binary"] = git_bin
    info["git_version"] = None
    if git_bin:
        rc, out, _ = _run([git_bin, "--version"])
        if rc == 0:
            info["git_version"] = out
    # Not initialized is a WARNING (we report, not error), not a hard failure.
    if not info.get("git_available"):
        info["status"] = WARNING
    elif not info.get("initialized"):
        info["status"] = WARNING
    else:
        info["status"] = PASS
    return info


def check_ffmpeg() -> Dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    status = PASS if (ffmpeg and ffprobe) else WARNING
    detail: Dict[str, Any] = {
        "status": status,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "note": (
            "ffmpeg/ffprobe are required for audio decoding in later stages; "
            "they are NOT installed automatically."
        ),
    }
    if ffmpeg:
        rc, out, _ = _run([ffmpeg, "-version"])
        detail["ffmpeg_version"] = out.splitlines()[0] if out else None
    return detail


def check_packages() -> Dict[str, Any]:
    versions = run_metadata.package_versions(CHECKED_PACKAGES)
    installed = {k: v for k, v in versions.items() if v != "not_installed"}
    missing = sorted(k for k, v in versions.items() if v == "not_installed")

    stage0_missing = [p for p in STAGE0_REQUIRED if p in missing]
    mainline_missing = [p for p in CPU_MAINLINE_REQUIRED if p in missing]

    if stage0_missing:
        status = FAIL
    elif mainline_missing:
        status = WARNING
    else:
        status = PASS

    return {
        "status": status,
        "versions": versions,
        "installed": installed,
        "missing": missing,
        "stage0_required_missing": stage0_missing,
        "cpu_mainline_required_missing": mainline_missing,
        "note": (
            "torch is intentionally NOT listed as mainline-required (GPU "
            "modules are optional). Missing packages are reported, not fixed."
        ),
    }


def check_gpu() -> Dict[str, Any]:
    """Inspect NVIDIA hardware and PyTorch CUDA. GPU absence is only WARNING."""
    nvidia_smi = shutil.which("nvidia-smi")
    info: Dict[str, Any] = {
        "status": WARNING,
        "nvidia_smi_available": bool(nvidia_smi),
        "nvidia_smi_path": nvidia_smi,
        "gpu_name": None,
        "gpu_memory_total_mib": None,
        "driver_version": None,
        "cuda_driver_version": None,
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda_available": None,
        "torch_cuda_version": None,
        "mixed_precision_possible": None,
        "device_capability": None,
        "note": (
            "GPU hardware presence does NOT imply PyTorch CUDA availability; "
            "both are reported independently."
        ),
    }

    # 1) hardware / driver via nvidia-smi
    if nvidia_smi:
        rc, out, _ = _run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        )
        if rc == 0 and out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 3:
                info["gpu_name"] = parts[0]
                try:
                    info["gpu_memory_total_mib"] = int(parts[1])
                except ValueError:
                    info["gpu_memory_total_mib"] = parts[1]
                info["driver_version"] = parts[2]
            rc2, out2, _ = _run([nvidia_smi], timeout=20)
            if rc2 == 0:
                m = re.search(r"CUDA Version:\s*([\d.]+)", out2)
                if m:
                    info["cuda_driver_version"] = m.group(1)

    # 2) PyTorch CUDA (independent of hardware detection)
    try:
        import torch  # type: ignore

        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        try:
            info["torch_cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover
            info["torch_cuda_available"] = False
            info["torch_cuda_error"] = str(exc)
        if info["torch_cuda_available"]:
            try:
                idx = torch.cuda.current_device()
                info["device_capability"] = list(
                    torch.cuda.get_device_capability(idx)
                )
                info["mixed_precision_possible"] = (
                    torch.cuda.get_device_capability(idx)[0] >= 7
                )
            except Exception:  # pragma: no cover
                info["mixed_precision_possible"] = None
    except Exception:
        info["torch_installed"] = False

    # Status logic: GPU is optional. Hardware present + torch CUDA ready => PASS.
    if info["torch_cuda_available"]:
        info["status"] = PASS
    elif info["gpu_name"]:
        info["status"] = WARNING  # hardware present, torch CUDA not usable
    else:
        info["status"] = WARNING  # no GPU at all
    return info


def check_directories(project_root: Path) -> Dict[str, Any]:
    """Check existence + permissions of key dirs. data/raw is read-checked only."""
    candidates = {
        "project_root": project_root,
        "data_raw": project_root / "data" / "raw",
        "data_interim": project_root / "data" / "interim",
        "features": project_root / "features",
        "reports": project_root / "reports",
        "logs": project_root / "logs",
        "config": project_root / "config",
    }
    results: Dict[str, Any] = {}
    failures: List[str] = []
    for key, path in candidates.items():
        exists = path.exists()
        rec = {
            "path": str(path.relative_to(project_root)) if key != "project_root" else ".",
            "exists": exists,
            "readable": os.access(str(path), os.R_OK) if exists else False,
            "writable": os.access(str(path), os.W_OK) if exists else False,
        }
        if key == "data_raw":
            rec["policy"] = "read_only_check_no_data_written"
            rec["writable"] = None  # do not assert writability of raw
        results[key] = rec
        if not exists and key != "data_raw":
            failures.append(key)
        elif key != "data_raw" and exists and not rec["readable"]:
            failures.append(key)
    status = FAIL if failures else PASS
    return {"status": status, "directories": results, "missing_or_unreadable": failures}


def check_config(project_root: Path) -> Dict[str, Any]:
    """Try to load+validate configuration; report PASS/WARNING/FAIL."""
    try:
        cfg = Config(project_root=project_root)
        summary = cfg.summary()
        return {
            "status": PASS,
            "config_dir": str(cfg.config_dir),
            "raw_data_read_only": summary["raw_data_read_only"],
            "all_paths_inside_root": summary["all_paths_inside_root"],
            "has_absolute_path_values": summary["has_absolute_path_values"],
            "n_path_keys": summary["n_path_keys"],
            "hard_constraints": summary["hard_constraints"],
            "error": None,
        }
    except ConfigError as exc:
        return {"status": FAIL, "error": str(exc)}
    except Exception as exc:  # pragma: no cover
        return {"status": FAIL, "error": f"{type(exc).__name__}: {exc}"}


def check_risks(
    os_info: Dict[str, Any],
    disk: Dict[str, Any],
    packages: Dict[str, Any],
    gpu: Dict[str, Any],
    ffmpeg: Dict[str, Any],
    git: Dict[str, Any],
    dirs: Dict[str, Any],
) -> Dict[str, Any]:
    risks: List[Dict[str, str]] = []

    if disk.get("free_gb") is not None and disk["free_gb"] < MIN_FREE_DISK_GB:
        risks.append(
            {
                "level": "WARNING",
                "area": "disk",
                "detail": (
                    f"Only {disk['free_gb']} GB free on project drive "
                    f"(recommended >= {MIN_FREE_DISK_GB} GB)."
                ),
            }
        )

    for pkg in packages.get("cpu_mainline_required_missing", []):
        risks.append(
            {
                "level": "WARNING",
                "area": "dependencies",
                "detail": (
                    f"CPU-mainline package '{pkg}' is missing; install before "
                    f"feature extraction / baseline modelling."
                ),
            }
        )
    for pkg in packages.get("stage0_required_missing", []):
        risks.append(
            {
                "level": "FAIL",
                "area": "dependencies",
                "detail": f"Stage-0 infrastructure package '{pkg}' is missing.",
            }
        )

    if not ffmpeg.get("ffmpeg") or not ffmpeg.get("ffprobe"):
        risks.append(
            {
                "level": "WARNING",
                "area": "audio_backend",
                "detail": "ffmpeg/ffprobe not found; audio decoding will fail later.",
            }
        )

    if gpu.get("gpu_name") and not gpu.get("torch_cuda_available"):
        risks.append(
            {
                "level": "WARNING",
                "area": "gpu",
                "detail": (
                    f"GPU hardware present ({gpu.get('gpu_name')}) but PyTorch "
                    f"CUDA is not usable "
                    f"(torch_installed={gpu.get('torch_installed')}). GPU "
                    f"modules will skip; CPU main line is unaffected."
                ),
            }
        )
    if not gpu.get("gpu_name"):
        risks.append(
            {
                "level": "WARNING",
                "area": "gpu",
                "detail": "No NVIDIA GPU detected via nvidia-smi; GPU modules are optional.",
            }
        )

    if not git.get("initialized"):
        risks.append(
            {
                "level": "WARNING",
                "area": "git",
                "detail": (
                    "Git repository is not initialized (no valid commit). "
                    "Run `git init` and make an initial commit to enable "
                    "code-version traceability."
                ),
            }
        )

    if dirs.get("missing_or_unreadable"):
        risks.append(
            {
                "level": "FAIL",
                "area": "directories",
                "detail": (
                    "Missing or unreadable project directories: "
                    + ", ".join(dirs["missing_or_unreadable"])
                ),
            }
        )

    if os_info.get("project_root_non_ascii"):
        risks.append(
            {
                "level": "WARNING",
                "area": "path_encoding",
                "detail": (
                    "Project root path contains non-ASCII characters; some "
                    "third-party libraries (e.g. audio backends) may misbehave."
                ),
            }
        )

    has_fail = any(r["level"] == "FAIL" for r in risks)
    status = FAIL if has_fail else (WARNING if risks else PASS)
    return {"status": status, "risks": risks}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_STATUS_RANK = {FAIL: 0, WARNING: 1, PASS: 2, NA: 3}


def overall_status(checks: Dict[str, Dict[str, Any]]) -> str:
    statuses = [c.get("status") for c in checks.values() if isinstance(c, dict)]
    statuses = [s for s in statuses if s in _STATUS_RANK]
    if not statuses:
        return NA
    worst = min(statuses, key=lambda s: _STATUS_RANK[s])
    if worst == FAIL:
        return FAIL
    if worst == WARNING:
        return WARNING
    return PASS


# ---------------------------------------------------------------------------
# Report assembly + IO
# ---------------------------------------------------------------------------

def resolve_paths(config_arg: Optional[str]) -> Tuple[Path, Path]:
    """Return (project_root, config_dir) from --config or auto-detection."""
    if config_arg:
        p = Path(config_arg).resolve()
        if p.is_file():  # a yaml file
            config_dir = p.parent
        else:  # a directory
            config_dir = p
        project_root = config_dir.parent
    else:
        project_root = default_project_root()
        config_dir = project_root / "config"
    return project_root, config_dir


def build_report(project_root: Path, argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run all checks and return the full report dict (no file IO)."""
    # Configuration-driven run metadata (config hash is deterministic).
    config_texts: Dict[str, str] = {}
    for name in ("paths.yaml", "config.yaml", "experiments.yaml"):
        fpath = project_root / "config" / name
        try:
            config_texts[name] = fpath.read_text(encoding="utf-8")
        except Exception:
            config_texts[name] = ""
    meta = run_metadata.build_run_metadata(
        project_root=project_root,
        config_texts=config_texts,
        argv=argv,
    )
    run_metadata.assert_no_absolute_raw_path(meta)

    os_info = check_os(project_root)
    checks: Dict[str, Dict[str, Any]] = {
        "os": os_info,
        "python": check_python(),
        "hardware": check_hardware(),
        "disk": check_disk(project_root),
        "git": check_git(project_root),
        "ffmpeg": check_ffmpeg(),
        "packages": check_packages(),
        "gpu": check_gpu(),
        "directories": check_directories(project_root),
        "config": check_config(project_root),
    }
    checks["risks"] = check_risks(
        os_info,
        checks["disk"],
        checks["packages"],
        checks["gpu"],
        checks["ffmpeg"],
        checks["git"],
        checks["directories"],
    )

    return {
        "run_metadata": meta,
        "overall_status": overall_status(checks),
        "checks": checks,
    }


def write_reports(
    report: Dict[str, Any],
    reports_dir: Path,
    run_id_override: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Write setup_check.json and setup_check.md into reports_dir."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id_override or report["run_metadata"]["run_id"]
    report["run_metadata"]["run_id"] = run_id

    json_path = reports_dir / "setup_check.json"
    md_path = reports_dir / "setup_check.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Markdown rendering (Chinese primary, English terms accurate)
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    return {
        PASS: "✅ PASS",
        WARNING: "⚠️ WARNING",
        FAIL: "❌ FAIL",
        NA: "— NOT APPLICABLE",
    }.get(status, status)


def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["run_metadata"]
    checks = report["checks"]
    lines: List[str] = []
    add = lines.append

    add("# 阶段 0 环境检查报告（setup_check）")
    add("")
    add(f"- **总体状态：** {_status_badge(report['overall_status'])}")
    add(f"- **Run ID：** `{meta['run_id']}`")
    add(f"- **UTC 时间：** {meta['utc_time']}")
    add(f"- **本地时间：** {meta['local_time']}（{meta['timezone']}）")
    add(f"- **配置哈希（config_hash）：** `{meta['config_hash']}`")
    add(f"- **配置文件：** {', '.join(meta['config_files'])}")
    add(f"- **命令行：** `{' '.join(shlex.quote(a) for a in meta['argv'])}`")
    add("")
    add("> 说明：GPU 不可用至多为 WARNING，绝不阻断 CPU 主流程。本检查不下载、"
       "不安装、不升级驱动/CUDA，也不写入 `data/raw` 任何数据。")
    add("")
    add("---")
    add("")

    # Check status summary table
    add("## 1. 检查项总览")
    add("")
    add("| 检查项 | 状态 |")
    add("| --- | --- |")
    order = [
        "os", "python", "hardware", "disk", "git", "ffmpeg",
        "packages", "gpu", "directories", "config", "risks",
    ]
    for key in order:
        if key in checks:
            add(f"| {key} | {_status_badge(checks[key].get('status'))} |")
    add("")

    def block(title: str, key: str, rows: List[Tuple[str, Any]]) -> None:
        add(f"## {title}")
        add("")
        add(f"**状态：** {_status_badge(checks[key].get('status'))}")
        add("")
        add("| 字段 | 值 |")
        add("| --- | --- |")
        for label, value in rows:
            add(f"| {label} | `{value}` |")
        add("")

    os_c = checks["os"]
    block("2. 操作系统与目录", "os", [
        ("system", os_c.get("system")),
        ("release / version", f"{os_c.get('release')} / {os_c.get('version')}"),
        ("machine", os_c.get("machine")),
        ("PROCESSOR_IDENTIFIER", os_c.get("processor_identifier")),
        ("python_platform", os_c.get("python_platform")),
        ("cwd", os_c.get("cwd")),
        ("project_root", os_c.get("project_root")),
        ("project_root 含非 ASCII", os_c.get("project_root_non_ascii")),
    ])

    py_c = checks["python"]
    block("3. Python 运行环境", "python", [
        ("executable", py_c.get("executable")),
        ("version", py_c.get("version")),
        ("implementation", py_c.get("implementation")),
        ("in_venv", py_c.get("in_venv")),
        ("conda_env", py_c.get("conda_env")),
        ("prefix / base_prefix", f"{py_c.get('prefix')} / {py_c.get('base_prefix')}"),
    ])

    hw = checks["hardware"]
    block("4. CPU 与内存", "hardware", [
        ("processor", hw.get("processor")),
        ("logical_cores", hw.get("logical_cores")),
        ("physical_cores", hw.get("physical_cores")),
        ("memory_total_gb", hw.get("memory_total_gb")),
        ("memory_available_gb", hw.get("memory_available_gb")),
    ])

    dk = checks["disk"]
    block("5. 磁盘空间（项目所在盘）", "disk", [
        ("drive", dk.get("drive")),
        ("total_gb", dk.get("total_gb")),
        ("used_gb", dk.get("used_gb")),
        ("free_gb", dk.get("free_gb")),
        ("min_recommended_free_gb", dk.get("min_recommended_free_gb")),
    ])

    gt = checks["git"]
    block("6. Git", "git", [
        ("git_available", gt.get("git_available")),
        ("git_version", gt.get("git_version")),
        ("initialized", gt.get("initialized")),
        ("commit", gt.get("commit")),
        ("branch", gt.get("branch")),
        ("dirty", gt.get("dirty")),
        ("error", gt.get("error") or "—"),
    ])

    ff = checks["ffmpeg"]
    block("7. ffmpeg / ffprobe", "ffmpeg", [
        ("ffmpeg", ff.get("ffmpeg") or "未找到"),
        ("ffprobe", ff.get("ffprobe") or "未找到"),
        ("ffmpeg_version", ff.get("ffmpeg_version") or "—"),
        ("note", ff.get("note")),
    ])

    # Packages detail
    pk = checks["packages"]
    add("## 8. Python 关键包")
    add("")
    add(f"**状态：** {_status_badge(pk.get('status'))}")
    add("")
    add("| 包 | 版本 |")
    add("| --- | --- |")
    for name in CHECKED_PACKAGES:
        add(f"| {name} | `{pk['versions'].get(name, 'not_installed')}` |")
    add("")
    if pk.get("missing"):
        add(f"- **缺失：** {', '.join(pk['missing'])}")
    if pk.get("stage0_required_missing"):
        add(f"- **阶段 0 必需但缺失：** {', '.join(pk['stage0_required_missing'])}")
    if pk.get("cpu_mainline_required_missing"):
        add(
            f"- **CPU 主流程必需但缺失（后续阶段前需安装）：** "
            f"{', '.join(pk['cpu_mainline_required_missing'])}"
        )
    add(f"\n> {pk.get('note')}\n")

    # GPU detail
    gpu = checks["gpu"]
    add("## 9. GPU / CUDA / PyTorch")
    add("")
    add(f"**状态：** {_status_badge(gpu.get('status'))}")
    add("")
    add("| 字段 | 值 |")
    add("| --- | --- |")
    for label, value in [
        ("nvidia_smi_available", gpu.get("nvidia_smi_available")),
        ("gpu_name", gpu.get("gpu_name") or "未检测到"),
        ("gpu_memory_total_mib", gpu.get("gpu_memory_total_mib")),
        ("driver_version", gpu.get("driver_version")),
        ("cuda_driver_version", gpu.get("cuda_driver_version")),
        ("torch_installed", gpu.get("torch_installed")),
        ("torch_version", gpu.get("torch_version") or "—"),
        ("torch_cuda_available", gpu.get("torch_cuda_available")),
        ("torch_cuda_version", gpu.get("torch_cuda_version") or "—"),
        ("device_capability", gpu.get("device_capability")),
        ("mixed_precision_possible", gpu.get("mixed_precision_possible")),
    ]:
        add(f"| {label} | `{value}` |")
    add(f"\n> {gpu.get('note')}\n")

    # Directories
    dr = checks["directories"]
    add("## 10. 目录可读写性")
    add("")
    add(f"**状态：** {_status_badge(dr.get('status'))}")
    add("")
    add("| 目录 | exists | readable | writable |")
    add("| --- | --- | --- | --- |")
    for key, rec in dr["directories"].items():
        add(
            f"| {key} (`{rec['path']}`) | {rec['exists']} | "
            f"{rec['readable']} | {rec['writable']} |"
        )
    add("")
    add("> `data/raw` 仅做存在与权限检查，本阶段不写入任何数据。")
    add("")

    # Config
    cf = checks["config"]
    add("## 11. 配置加载与校验")
    add("")
    add(f"**状态：** {_status_badge(cf.get('status'))}")
    add("")
    if cf.get("error"):
        add(f"```\n{cf['error']}\n```")
    else:
        add("| 字段 | 值 |")
        add("| --- | --- |")
        add(f"| config_dir | `{cf.get('config_dir')}` |")
        add(f"| raw_data_read_only | {cf.get('raw_data_read_only')} |")
        add(f"| all_paths_inside_root | {cf.get('all_paths_inside_root')} |")
        add(f"| has_absolute_path_values | {cf.get('has_absolute_path_values')} |")
        add(f"| n_path_keys | {cf.get('n_path_keys')} |")
        add(f"| hard_constraints | `{cf.get('hard_constraints')}` |")
    add("")

    # Risks
    rk = checks["risks"]
    add("## 12. 环境风险汇总")
    add("")
    add(f"**状态：** {_status_badge(rk.get('status'))}")
    add("")
    if rk.get("risks"):
        add("| 级别 | 领域 | 说明 |")
        add("| --- | --- | --- |")
        for r in rk["risks"]:
            add(f"| {r['level']} | {r['area']} | {r['detail']} |")
    else:
        add("未发现风险。")
    add("")

    # Conclusion
    add("---")
    add("")
    add("## 13. 结论")
    add("")
    overall = report["overall_status"]
    if overall == PASS:
        add("环境满足阶段 0 要求，CPU 主流程可运行。")
    elif overall == WARNING:
        add("环境基本满足阶段 0 要求（存在 WARNING），CPU 主流程不受阻断；"
            "进入后续阶段前需解决相关依赖/工具缺失。")
    else:
        add("存在 FAIL 级别问题，需先修复后再继续。")
    add("")
    add("- 本报告由 `scripts/00_setup_check.py` 生成。")
    add("- 未下载、未读取或处理任何真实患者数据。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 0 environment check.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the config directory or a config yaml file "
        "(default: auto-detected <project_root>/config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks and print to console only; do NOT write report files.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the run id (default: <utc>-<config_hash[:8]>).",
    )
    args = parser.parse_args(argv)

    project_root, config_dir = resolve_paths(args.config)
    logger = get_logger(
        "setup_check",
        log_dir=None if args.dry_run else project_root / "logs",
        run_id=args.run_id,
        file_name="setup_check.log",
    )
    logger.info("Starting Stage 0 environment check (dry_run=%s)", args.dry_run)
    logger.info("project_root=%s", project_root)
    logger.info("config_dir=%s", config_dir)

    report = build_report(project_root, argv=sys.argv)
    if args.run_id:
        report["run_metadata"]["run_id"] = args.run_id

    logger.info("Overall status: %s", report["overall_status"])

    if args.dry_run:
        logger.info("Dry-run: reports were NOT written to disk.")
        logger.info("Run without --dry-run to write reports/setup_check.{json,md}")
        print(json.dumps({"overall_status": report["overall_status"],
                          "run_id": report["run_metadata"]["run_id"]},
                         indent=2, ensure_ascii=False))
        return 0

    reports_dir = project_root / "reports"
    json_path, md_path = write_reports(
        report, reports_dir, run_id_override=args.run_id
    )
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)

    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "run_id": report["run_metadata"]["run_id"],
                "setup_check_json": str(json_path),
                "setup_check_md": str(md_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    # Non-zero exit only on FAIL so CI/automation can gate.
    return 0 if report["overall_status"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
