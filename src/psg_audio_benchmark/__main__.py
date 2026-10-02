"""Minimal, testable console entry point for ``psg_audio_benchmark``.

This module exists to satisfy the ``[project.scripts]`` declaration in
``pyproject.toml``::

    psgb-setup-check = "psg_audio_benchmark.__main__:main"

Phase 0 declared that entry but never created ``__main__.py`` (a leftover
interface defect caught at the start of Phase 1, see prompt section 0.1). This
file is an *engineering-completeness* fix, NOT data processing: it only prints
project identity and run-mode info and returns an exit code. It deliberately
does not download data, read patient files, parse annotations or build models,
so it cannot enlarge Phase 1 scope.

Run as a module or installed console script::

    python -m psg_audio_benchmark            # default info banner
    python -m psg_audio_benchmark --info     # same, explicit
    python -m psg_audio_benchmark --version  # package version
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .config import Config, default_project_root

#: Human-readable description of what this entry does and does NOT do.
NON_CLINICAL_NOTE = (
    "Public-dataset benchmark study only. Not a clinically deployable "
    "diagnostic system, not a PSG replacement, not an external clinical "
    "validation, and not an upper-airway obstruction localization model."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psg_audio_benchmark",
        description=(
            "PSG_Audio_Benchmark console entry (engineering smoke entry). "
            "Prints project identity and run-mode info only."
        ),
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print project identity, configuration summary and exit (default).",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    return parser


def _print_info(stream=None) -> None:
    # Resolve sys.stdout at CALL time (not as a default arg) so test capture
    # (capsys) and runtime redirection both see the output.
    out_stream = stream if stream is not None else sys.stdout
    lines: List[str] = []
    lines.append(f"PSG_Audio_Benchmark v{__version__}")
    lines.append(NON_CLINICAL_NOTE)
    try:
        root = default_project_root()
        lines.append(f"project_root (resolved): {root}")
        cfg = Config(project_root=root)
        summary = cfg.summary()
        lines.append(
            "hard_constraints: patient_level_split_required="
            f"{summary['hard_constraints']['patient_level_split_required']}, "
            f"raw_data_read_only={summary['raw_data_read_only']}, "
            f"feature_format={summary['hard_constraints']['feature_format']}"
        )
        lines.append(
            "primary dataset DOI: "
            f"{cfg.config_data['dataset']['primary']['data_doi']} "
            "(literature-reported numbers are hypotheses pending audit)"
        )
    except Exception as exc:  # pragma: no cover - defensive only
        lines.append(f"configuration summary unavailable: {exc}")
    print("\n".join(lines), file=out_stream)


def main(argv: Optional[List[str]] = None) -> int:
    """Console entry. Returns a process exit code (0 on success).

    Kept intentionally tiny so it is safe to call from tests and from the
    installed ``psgb-setup-check`` script.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    # --info or no subcommand: print the info banner.
    _print_info()
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution path
    raise SystemExit(main())
