"""Stage 15b — BSPC pre-submission revision re-analysis package.

Implements the duration-overlap label revision, the model/feature/lag battery and the
multi-level (window / patient-macro / event-level) evaluation on the INHERITED frozen v2
patient split, with strict fold-internal leakage control. Stage 2-14 products are never
modified; all outputs are isolated under the Stage-15b run dir. Audio stays BLOCKED.
"""

from . import labels, features, nested_cv, eval_metrics  # noqa: F401

__all__ = ["labels", "features", "nested_cv", "eval_metrics"]
