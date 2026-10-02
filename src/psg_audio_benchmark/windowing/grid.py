"""Pure window-grid generation for Stage 4.

Generates the candidate, non-overlapping, half-open ``[start, end)`` windows for
one patient over its core-signal-intersection domain ``[domain_start, domain_end)``.

The grid origin is the intersection start; windows step by ``hop_step_s`` while
their start is strictly below the intersection end. The last candidate window
may extend past the intersection end -- that tail window is a genuine candidate
that the coverage step will mark ``excluded_incomplete_core_signal_coverage``
(never zero-padded or truncated to fake completeness).

Nothing here reads files or touches raw data.
"""

from __future__ import annotations

from typing import Iterator, Tuple


def generate_window_grid(
    domain_start: float,
    domain_end: float,
    window_length_s: float,
    hop_step_s: float,
) -> Iterator[Tuple[int, float, float]]:
    """Yield ``(window_index, start_rel, end_rel)`` for one patient's domain.

    Parameters mirror the resolved config. ``window_index`` is 0-based per
    patient. Generation stops once ``start_rel >= domain_end``. If the domain is
    empty or invalid (``domain_end <= domain_start``), nothing is yielded.
    """
    if window_length_s <= 0:
        raise ValueError("window_length_s must be positive")
    if hop_step_s <= 0:
        raise ValueError("hop_step_s must be positive")
    if domain_end <= domain_start:
        return
    start = float(domain_start)
    end_domain = float(domain_end)
    idx = 0
    # Step while the window START is inside the domain. The window END may
    # exceed domain_end (tail); coverage handles that honestly.
    while start < end_domain:
        yield idx, start, start + window_length_s
        start += hop_step_s
        idx += 1


__all__ = ["generate_window_grid"]
