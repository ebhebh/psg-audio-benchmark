# Third-party licenses (runtime and dev dependencies)

**LOCAL_CANDIDATE_NOT_PUBLISHED**

All third-party components are used via normal package imports; **no
third-party source code is vendored or copied** into this repository. Each
package retains its own license; nothing in this release relicenses them.
Versions and license statements below were read from the installed
distributions' metadata (environment used for release verification: Windows 10,
CPython 3.13.9). In any conflict, the upstream package's license text is
authoritative.

| Package | Verified version | License (from installed metadata) |
|---|---|---|
| numpy | 2.3.5 | BSD (NumPy Developers) |
| pandas | 2.3.3 | BSD 3-Clause |
| scikit-learn | 1.7.2 | BSD 3-Clause (new BSD); installed metadata carries the full text upstream |
| pyarrow | 21.0.0 | Apache Software License (Apache 2.0) |
| scipy | 1.16.3 | BSD (SciPy developers) |
| matplotlib | 3.10.6 | Matplotlib license (PSF-style for ≥1.3.0; based on PSF License) |
| PyYAML | 6.0.3 | MIT |
| pytest (dev) | 8.4.2 | MIT |

Optional (not installed during verification; lazily imported, availability-checked):

| Package | Used for | License |
|---|---|---|
| py7zr ≥0.20 | inspecting/extracting 7z staged archives only | LGPL-3.0-or-later (check the version you install) |
| soundfile ≥0.12 | audio metadata probing only (audio analysis BLOCKED, never runs) | BSD 3-Clause |

## Dataset-derived material

`results_published/`, `figures_published/`, `tables_published/` derive from the
Tao et al. 2025 dataset — Data DOI `10.57760/sciencedb.19070`, version V5,
**CC BY 4.0** on the data page. Attribution requirements of CC BY 4.0 apply to
reuse of those derived aggregates independently of whatever code license is
eventually chosen (see `../LICENSE_DECISION_REQUIRED.md`).

## Fonts

`figures_published/` embeds standard figure fonts (PDF fonttype 42). The
rendering style references Times New Roman; no font file is redistributed here.
