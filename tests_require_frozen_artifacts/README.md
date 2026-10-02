# Tests that require the authors' frozen pipeline artifacts or the real dataset

These test files are NOT collected by the default test run (`pytest tests`,
see pyproject `testpaths`). They audit FROZEN intermediate products
(`data/raw`, real window indexes, the approved v2 split run outputs, phase-02
erratum files) that exist only after the pipeline has been executed on the
official dataset — or that are pinned to the authors' environment.

They are shipped unmodified for transparency. Expect FAILURES (not skips) when
the referenced artifacts are absent; this is by design and does NOT indicate a
code defect. After a full Mode-B pipeline run (see docs/REPRODUCTION_GUIDE.md)
most of them become runnable against your own artifacts, except those pinned to
specific historical run ids.

Moved here (unmodified) from tests/ during release packaging:
- test_stage3_raw_immutability.py   (1 real-data test + 1 fixture test)
- test_stage4_cross_midnight_erratum.py (case8 real-window tests + erratum-file test)
- test_stage6b_gate.py              (1 real-output test + 7 synthetic gate-logic tests)
- test_stage6b_production_audit.py  (audit of the approved v2 split run outputs)
