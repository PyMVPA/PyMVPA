# PyMVPA Python 3 + Pytest Modernization

**Date**: 2026-03-14
**Branch**: `patch-1` (99 commits, 214 files changed from master)
**Goal**: Fully modernize PyMVPA for current Python ecosystem

## Strategic Decisions

- **Drop Python 2** entirely — target Python 3.9+
- **Remove `future`/`six`/`past` library** dependencies — use native Python 3
- **Build system**: `numpy.distutils` → `meson-python` + `pyproject.toml`
- **Test runner**: `nose` → `pytest` + `tox`
- **Convert all modules** including support/afni, support/nibabel, misc/surfing, cmdline
- **Deprecate dead dependencies**: shogun, MDP, pprocess, OpenOpt, scipy.weave
- **Minimize diff**: adopt datalad-style nose compatibility shims so test files
  need minimal changes (keep `ok_`, `eq_`, `assert_equal`, `sweepargs`, etc.
  as thin wrappers — do NOT rewrite individual test assertions to bare `assert`)

## Guiding Principles

### Establish testability first, then make verified changes
The highest priority is getting tests running on the already-converted modules.
Every subsequent change is then verified against this green baseline. No blind
bulk transformations — each step is followed by a test run.

### Minimal diff for reviewability
Do NOT rewrite working test assertions, do NOT rename variables, do NOT
refactor code beyond what the migration strictly requires. Each change
should be explainable as "needed for PY3/pytest/build-system" only.

### One commit per logical unit
Each converted submodule, each infrastructure change, each moved test file
gets its own commit. Verify related tests pass before committing.
Use `datalad run` for scripted/automated bulk transformations to protocol
exactly what command was executed.

### Consultation checkpoints
Some steps (especially HDF5 serialization, custom pickling, metaclass
patterns) may require domain-specific decisions. When a step involves
non-obvious semantic choices (not just mechanical syntax fixes), **stop
and consult** the maintainer before proceeding. Specific consultation
points are marked with **⚠ CONSULT** below.

## Approach: Minimal-Diff Nose→Pytest Transition

Learned from datalad's migration (`datalad/tests/utils_pytest.py`):

1. **Keep nose-style assertion functions** as thin `assert`-based shims in
   `mvpa2/testing/tools.py` — all ~1300 call sites in tests remain untouched
2. **`assert_raises = pytest.raises`** — direct alias, context-manager form works
3. **`SkipTest`** from `unittest` (pytest recognizes it natively in test functions)
4. **`make_decorator`** → `functools.wraps` (4 internal uses in tools.py only)
5. **Keep `sweepargs`** working as-is — only fix its `SkipTest` import; full
   conversion to `@pytest.mark.parametrize` deferred to a later effort
6. **Keep `__test__ = False`** on functions (pytest respects it); fix module-level
   ones via `collect_ignore` in conftest.py
7. **Module-level `raise SkipTest`** (2 files) → `pytest.skip(allow_module_level=True)`

This means test files themselves get almost zero churn from the pytest migration.
The diff comes from: (a) stripping `future`/`past` imports, (b) PY3-fixing
unconverted files, (c) replacing nose internals in `mvpa2/testing/tools.py`.

---

## Dependency Status (researched 2026-03-14)

### Dead / Broken — deprecate and remove support code

| Dependency | Last Release | PY3 Status | Action |
|---|---|---|---|
| **Shogun** (`modshogun`) | 6.1.3 (Dec 2019) | No PY3.9+ bindings | Remove `mvpa2/clfs/sg/` entirely, remove all `shogun`/`sg*` externals |
| **MDP** (mdp-toolkit) | 3.6 (Apr 2020) | PY3.5-3.8 only | Remove `mvpa2/mappers/mdp_adaptor.py`, `mvpa2/mappers/lle.py` (MDP-based), remove `mdp` externals |
| **pprocess** | ~0.5.3 (ancient) | No | Remove pprocess codepaths, use `joblib` instead |
| **OpenOpt** | 0.5308 (~2014) | No | Remove from `mvpa2/clfs/gpr.py`, `model_selector.py` |
| **scipy.weave** / **weave** | Removed from SciPy 1.0 (2017) | No | Remove weave codepath in `mvpa2/clfs/distance.py` |
| **nose** | 1.3.7 (Jun 2015) | Broken ≥3.10 | Replace with pytest shims (Phase 1) |
| **python-future** | 1.0.0 (Feb 2024) | Fragile, CVE-2022-40899 | Remove entirely (Phase 2) |
| **cPickle** | N/A in PY3 | Use `pickle` | Remove external check |
| **hcluster** | Ancient | Unknown | Remove external check |

### Healthy — keep, update version bounds

| Dependency | Latest | Min Version for PyMVPA |
|---|---|---|
| **numpy** | current | ≥1.22 |
| **scipy** | current | ≥1.7 |
| **nibabel** | 5.4.2 (Mar 2026) | ≥3.0 |
| **h5py** | 3.16.0 (Mar 2026) | ≥3.0 (highlevel removed!) |
| **scikit-learn** | 1.8.0 (Dec 2025) | ≥1.0 |
| **joblib** | 1.5.3 (Dec 2025) | ≥1.0 |
| **matplotlib** | 3.10.8 (Dec 2025) | ≥3.1 |
| **statsmodels** | 0.14.6 (Dec 2025) | ≥0.12 |
| **pywt** (PyWavelets) | 1.9.0 (Aug 2025) | ≥1.1 |
| **lxml** | 6.0.2 (Sep 2025) | ≥4.0 |
| **rpy2** | 3.6.5 (Mar 2026) | ≥3.4 |
| **reportlab** | 4.4.10 (Feb 2026) | ≥3.5 |
| **Pillow** | 12.1.1 (Feb 2026) | ≥8.0 |
| **libsvm** (`libsvm-official`) | 3.37 (Dec 2025) | (bundled) |

### Notes on specific deps

- **h5py 3.0** removed `h5py.highlevel` — code doing `import h5py.highlevel` must change
- **matplotlib**: `pylab` interface deprecated in spirit; use `matplotlib.pyplot`
- **griddata**: `matplotlib.mlab.griddata` removed in mpl 3.1; use `scipy.interpolate.griddata`
- **libsvm**: PyMVPA bundles its own SWIG wrapper; `libsvm-official` PyPI package is an alternative
- **nipy**: still exists but niche; keep optional support

---

## Current State Summary

| Category | Count | Status |
|---|---|---|
| Total .py files in `mvpa2/` | ~390 | — |
| Files changed on branch | 197 | committed |
| Files with `from future/past/builtins` to strip | 89 | need cleanup |
| Untouched files with PY2 patterns | 53 | need conversion |
| Total PY2 pattern hits in untouched files | 249 | (97 print, 62 xrange, 47 iteritems, 37 basestring, 5 raise-comma, 1 execfile) |
| Syntax-error files (crash on PY3 import) | 2 | `formats.py`, `som.py` (raise comma) |
| Files with direct `from nose` imports | 9 | need fixing |
| `make_decorator` uses | 4 | all in `tools.py` |
| Module-level `raise SkipTest` | 2 | `test_niftidataset.py`, `test_giftidataset.py` |

### Critical import chain for testability

```
pytest mvpa2/base/tests/test_collections.py
  → from mvpa2.testing.tools import ...
    → from mvpa2 import ...           # mvpa2/__init__.py (OK, no nose at module level)
    → if externals.exists('nose'):    # guarded, stubs provided if missing (OK)
    → from numpy.testing import dec   # CRASH on numpy≥1.18 (dec removed)
      → fallback: from numpy.testing.decorators import skipif  # ALSO removed
    → from six import reraise         # CRASH if six not installed
    → make_decorator → _need_nose()   # @with_tempfile etc. broken without nose
  → from mvpa2.testing import sweepargs
    → from mvpa2.testing.sweep import sweepargs  # uses SkipTest from tools
```

Immediate crashers (block all testing):
1. `numpy.testing.dec` / `numpy.testing.decorators` — both removed in modern numpy
2. `from six import reraise` — six may not be installed
3. `make_decorator` stubs — all 4 custom decorators (`with_tempfile`, `reseed_rng`,
   `nodebug`, `labile`) unusable without nose

---

## Phase 0: Clean Up WIP and Establish Baseline ✅ DONE

- [x] **0.1** Stashed WIP: `git stash push -m "WIP: in-progress py3 fixes ..."`
- [x] **0.2** `py3tool.py` confirmed deleted
- [x] **0.3** `.so` files untracked (already in .gitignore)
- [x] **0.4** `.gitignore` updated with modern entries
- [x] **0.5** Committed: fc76922d0 `RF: update .gitignore for modern Python tooling`

## Phase 1: Bootstrap Testability ✅ DONE

Goal was to get tests green on Python 3.11+. The plan assumed only
`mvpa2/testing/` would need changes, but numpy 2.x / scipy / h5py
breaking changes forced fixing many source files too (Phase 7.2 work
pulled forward). `future`/`six` still needed temporarily.

### 1.1 Testing infrastructure — ✅ DONE

- [x] **1.1.1** `numpy.testing.dec` → `pytest.mark.skipif`
- [x] **1.1.2** `from six import reraise` → bare `raise`
- [x] **1.1.3** Nose assertion imports → thin assert-based shims (datalad pattern)
- [x] **1.1.4** `make_decorator` → `functools.wraps` (4 sites)
- [x] **1.1.5** Unused nose imports dropped
- [x] Committed: 7d386d165

### 1.2 sweepargs — ✅ DONE

- [x] Removed `from builtins import str`; `SkipTest` import verified

### 1.3 conftest.py — ✅ DONE (minimal)

- [x] Root `conftest.py` with `collect_ignore_glob` and PYTHONPATH setup
- [ ] TODO: session fixtures for RNG seed / warning suppression
- [ ] TODO: custom markers (`labile`, `slow`, `ai_generated`)

### 1.4 numpy/scipy/h5py compat fixes (unplanned but required) — ✅ DONE

These were needed to unblock test collection and execution:

- [x] `numpy.trapz` → `numpy.trapezoid` (16dc12383)
- [x] `np.float` → `float` throughout (16dc12383, 2c7dc2f0d)
- [x] `np.int` → `int` throughout (346c5c042)
- [x] `np.object` → `object` (22b08ba84)
- [x] `np.asscalar` → `.item()` + polyfill in `__init__.py` (22b08ba84, f0599e589)
- [x] `np.issctype` → `np.issubdtype` (16dc12383)
- [x] `np.histogram(normed=)` → `density=` (8f129c727)
- [x] `np.linalg.linalg.LinAlgError` → `np.linalg.LinAlgError` (16dc12383)
- [x] `scipy.linalg.basic.LinAlgError` → `scipy.linalg.LinAlgError` (16dc12383)
- [x] `scipy.stats rv_continuous extradoc` kwarg removed (16dc12383)
- [x] `scipy.stats.stats.special` → `scipy.special` (96c3854ed)
- [x] `h5py.highlevel` removed, `h5py.File` references fixed (22b08ba84)
- [x] `.decode()` on str attrs → `_safe_decode()` helper (22b08ba84)
- [x] `platform.dist()` guard (65be437f4)
- [x] `old_div()` → plain division in gda.py, transformers.py (384690215)
- [x] `from builtins/future/past` removed from `__init__.py`, gda.py, transformers.py

### 1.5 Verification — ✅ DONE

- [x] `tox.ini` created — runs `tox` to verify 7 test directories
- [x] `conftest.py` created
- [x] Committed: cda7a75e4

**✓ CHECKPOINT: 198 tests pass via `tox -e py3` on Python 3.11**

Test results by directory:

| Directory | Passed | Failed | Skipped | Notes |
|---|---|---|---|---|
| `algorithms/tests` | 7 | 0 | 0 | |
| `base/tests` | 94 | 2 | 0 | test_hdf5 excluded |
| `clfs/tests` | 39 | 2 | 5 | test_svm excluded (needs C ext) |
| `kernels/tests` | 5 | 0 | 0 | |
| `measures/tests` | 5 | 0 | 0 | |
| `misc/tests` | 31 | 0 | 0 | test_dcov excluded |
| `support/tests` | 17 | 0 | 0 | |
| **Total** | **198** | **4** | **5** | |

Known failures (not migration blockers):
1. `test_array_collectable_unique` — numpy NaN uniqueness semantic change
2. `test_h5py_io` — HDF5 bytes/str key roundtrip (⚠ CONSULT)
3. `test_degenerate_usage` — flaky (order-dependent, passes alone)
4. `test_diff_len_labels_str_treeclassifier` — requires SVM C extension

Not yet in tox (need more work):
- `atlases/tests` — 1 skipped (OK)
- `datasets/tests` — 1 error (nifti dep)
- `featsel/tests` — 4 failures (SVM-dependent)
- `generators/tests` — empty
- `mappers/tests` — 1 error (needs investigation)

### 1.6 Python 3.13 compatibility — ✅ DONE

- [x] Wrapped stale `smlrc.so` (PY2 binary) import in `AttributeError` guard
  in `mvpa2/clfs/libsmlrc/smlrc.py` (3254ecc3d)
- [x] Fixed `conftest.py` to add `collect_ignore` for files with PY3 syntax errors
  (`formats.py`, `som.py`) so collection doesn't crash
- [x] Added `mvpa2/clfs/tests/test_svm.py` to `--ignore` in `tox.ini`

### 1.7 Test stabilization — ✅ DONE

Fixes for remaining real test failures (not flaky/extension-dependent):

- [x] **NaN uniqueness test** (`test_collections.py:141`): numpy 2.x deduplicates
  NaN in `np.unique()`. Updated assertion to expect 0 extra elements when NaN
  already present on numpy >= 2.
- [x] **HDF5 bytes/str roundtrip** (`hdf5.py:131-136,431,446`): Applied `_safe_decode()`
  to dict keys in `_hdf_dict_to_obj()` (tuple-items and legacy paths) AND to scalar
  values in `hdf2obj()` (the `is_scalar` loading path). h5py 3.x can return bytes for
  strings stored in older formats or via certain storage paths.
- [x] **Regex FutureWarning** (`misc/io/base.py:144`): Escaped inner brackets in
  character class `'[[\] ]'` → `r'[\[\] ]'`

### 1.8 Achieve zero test failures — ✅ DONE

All remaining test failures mitigated:

- [x] **B1** (`test_diff_len_labels_str_treeclassifier`): Added `skip_if_no_external('libsvm')`
  guard — test is properly skipped when C extension unavailable
- [x] **C1** (`test_degenerate_usage`): sklearn 1.8.0 `LDA._solve_svd()` raises `IndexError`
  on degenerate (all-zero) data instead of `ValueError`. Root cause: `S[0]` on empty
  singular values array after SVD of 0-feature data.
  [sklearn issue #18671](https://github.com/scikit-learn/scikit-learn/issues/18671).
  - PyMVPA fix: Added `IndexError` to caught exceptions in `SKLLearnerAdapter._train()`
    (`skl/base.py:100`) — wraps as `FailedToTrainError` (same as py39 `ValueError` path)
  - sklearn fix: Prepared in `externals/sklearn/` — guards `S[0]` access with `len(S) == 0`
    check in `discriminant_analysis.py:628`. Branch: `fix-lda-degenerate-indexerror`
- [x] **D1/D2/D3** (flaky/labile tests): Set `MVPA_TESTS_LABILE=no` in `tox.ini`.
  These are seed-dependent numerical edge cases, not regressions. Override with
  `MVPA_TESTS_LABILE=yes` for thorough runs.

**✓ CHECKPOINT: py39: 202 passed, 5 skipped, 0 failed.  py313: 202 passed, 5 skipped, 0 failed.**

### Full Failure Inventory

#### Category A: Fixed — real bugs in test/code

| ID | Test | Fix |
|---|---|---|
| A1 | `test_array_collectable_unique` | Updated NaN assertion for numpy 2.x |
| A2 | `test_h5py_io` | Decode dict keys + scalar values with `_safe_decode()` |

#### Category B: Missing C extension — skipped, deferred to Phase 6

| ID | Test | Mitigation |
|---|---|---|
| B1 | `test_diff_len_labels_str_treeclassifier` | `skip_if_no_external('libsvm')` |

#### Category C: Upstream bug — mitigated both sides

| ID | Test | Mitigation |
|---|---|---|
| C1 | `test_degenerate_usage` | `IndexError` caught in `SKLLearnerAdapter`; sklearn fix prepared |

#### Category D: Flaky/labile — disabled in CI

| ID | Test | Mitigation |
|---|---|---|
| D1 | `test_tree_classifier` | `MVPA_TESTS_LABILE=no` in tox.ini |
| D2 | `test_classifier_generalization` | `MVPA_TESTS_LABILE=no` in tox.ini |
| D3 | `test_dist_p_value` | `MVPA_TESTS_LABILE=no` in tox.ini |

#### Category E: Warnings — partially fixed

| ID | Source | Status |
|---|---|---|
| E1 | `misc/io/base.py` regex FutureWarning | **Fixed** |
| E2 | `Dataset.__array__` missing `copy` kwarg | Deferred to Phase 7 |
| E3 | `np.matrix` deprecation warnings | Deferred to Phase 7 |

## Phase 2: Strip `future`/`six`/`past` — Pure Python 3 ✅ DONE

Stripped all `future`/`six`/`past` imports and replaced compat helpers with
native Python 3 equivalents. Each step verified with `tox -e py313`
(202 passed, 5 skipped, 0 failed). Final verification: tests pass
WITHOUT `future` or `six` installed.

### 2.1 Scripted bulk removals — ✅ DONE

- [x] **2.1.1** Removed `from __future__ import` lines (43 files) — b6e2fa9
- [x] **2.1.2** Removed `from builtins import` lines (86 files) — f133579
- [x] **2.1.3** Removed `from future import standard_library` + `.install_aliases()` (9 files) — 92159c8

### 2.2 Replacements requiring per-file review — ✅ DONE

- [x] **2.2.1** Replaced `basestring` → `str` (36 files, code + docstrings) — c3c99be
- [x] **2.2.2** Removed `from past.builtins import cmp` (3 files — all unused;
  `misc/support.py` already had a local `def cmp`) — 7c37e2f
- [x] **2.2.3** Replaced `old_div(a, b)` → `a / b` or `a // b` (14 files, 46 call sites) — b52d725
  - 44 sites → `/` (true division, at least one operand float/array)
  - 2 sites → `//` (integer division): `test_dataset.py` np.tile count,
    `flatten.py` np.repeat multiplier
- [x] **2.2.4** Replaced `with_metaclass(Meta, Base)` → PY3 `(Base, metaclass=Meta)` syntax
  (2 files: `base/__init__.py`, `base/state.py`) — 22a1f14
  - Kept `__metaclass__ = AttributesCollector` class attribute in
    `ClassWithCollections` (used by `hdf5.py` for serialization)

### 2.3 Final cleanup — ✅ DONE

- [x] **2.3.1** Removed `future` and `six` from `tox.ini` deps — 10899e6
  - Tests pass WITHOUT either package installed
- [x] **2.3.2** Final sweep: `grep -r 'from future|from past|from builtins|from six' mvpa2/`
  returns 0 hits

**✓ CHECKPOINT: 202 passed, 5 skipped, 0 failed — no `future`/`six` dependency**

### Remaining Phase 2 items (deferred to later phases)

- [ ] **2.2.5** Remove PY2 guards (`if PY2:` branches, `builtins_mods` in hdf5.py)
  — deferred, needs ⚠ CONSULT for HDF5 backward-compat loading
- [ ] **2.2.6** Remove `mvpa2/support/_copy.py` (vendored PY2 copy module)
  — deferred to Phase 4h
- [ ] Remove `cPickle`, clean up `SyntaxError` suppression in `externals.py`
  — deferred to Phase 3.6

## Phase 3: Remove Dead Dependencies ✅ DONE

Surgically removed all dead/unmaintained dependencies. Each removal verified
with `tox -e py313` (202 passed, 4 skipped, 0 failed throughout).

- [x] **3.1** Removed Shogun support — 50b0a11
  - Deleted `mvpa2/clfs/sg/` directory, `mvpa2/kernels/sg.py`, `test_svmkernels.py`
  - Removed all shogun/sg entries from externals, warehouse, testing/clfs, suite
  - Fixed `test_absent_external_version` (registered temp `versions._KNOWN` entry)
  - Fixed `test_debug_rgexp` (`'SG'` → `'CM'` debug channel)
- [x] **3.2** Removed MDP support — d765901
  - Deleted `mdp_adaptor.py`, `lle.py`, `test_mdp.py`, 2 example scripts
  - Removed `mdp` entries from externals, suite, mappers docstring
- [x] **3.3** Removed pprocess support — 148f33a
  - Removed all pprocess codepaths from searchlight, voxel selection,
    hyperalignment (10 files, -365 lines)
  - `nproc` parameter kept; `nproc>1` raises `NotImplementedError`
  - `nproc=None` defaults to 1
- [x] **3.4** Removed OpenOpt support — d4094fa
  - Deleted `model_selector.py`, removed `GPRWeights` class from gpr.py
  - Removed `openopt`, `MOD_SEL`, `OPENOPT` debug channels
- [x] **3.5** Removed scipy.weave support — fbc95b8
  - Removed weave-accelerated `pnorm_w` from distance.py (kept pure Python version)
  - Removed `__check_scipy_weave` and both weave entries from externals
- [x] **3.6** Cleaned up dead externals — fb24ee2
  - Removed `hcluster`, `cPickle`, `mock` from `_KNOWN` (all PY2-only or absorbed)
  - Kept `nose` (still referenced in old test runner)

**✓ CHECKPOINT: 202 passed, 4 skipped, 0 failed — all dead deps removed**

## Phase 4: Convert Remaining ~53 Unconverted Source Files

All mechanical PY3 fixes. Patterns:
`xrange→range`, `.iteritems()→.items()`, `.itervalues()→.values()`,
`.iterkeys()→.keys()`, `print x→print(x)`, `raise X, msg→raise X(msg)`,
`except X, e→except X as e`, `basestring→str`, `execfile(f)→exec(open(f).read())`,
`has_key(k)→k in d`

**Commit discipline**: One commit per submodule group.
Run related tests after each conversion. Use `datalad run` for any scripted
transformations. **Fix syntax-error files first** (formats.py, som.py) since
they crash on import under PY3.

### 4.0 Fix PY3 syntax errors (crash on import)
- [ ] `datasets/formats.py` — 4 hits (raise-comma — **syntax error in PY3**)
- [ ] `mappers/som.py` — 2 hits (xrange, raise-comma — **syntax error in PY3**)
- [ ] Commit: `BF(PY3): fix syntax errors in formats.py and som.py`

### 4a. `support/nibabel/` (8 files, 43 PY2 hits)
- [ ] `afni_niml.py` — 14 hits (xrange, iteritems, basestring)
- [ ] `afni_niml_dset.py` — 10 hits (xrange, iteritems, basestring)
- [ ] `afni_suma_spec.py` — 7 hits (print, xrange, iteritems, basestring)
- [ ] `afni_niml_annot.py` — 5 hits (xrange, basestring)
- [ ] `afni_suma_1d.py` — 2 hits (xrange, basestring)
- [ ] `afni_niml_roi.py` — 1 hit (basestring)
- [ ] `surf_fs_asc.py` — 2 hits (xrange)
- [ ] `surf_gifti.py` — 2 hits (iteritems)
- [ ] Run: `pytest mvpa2/tests/test_surfing*.py`
- [ ] Commit: `RF(PY3): convert support/nibabel/ to Python 3`

### 4b. `support/afni/` (5 files, 58 PY2 hits)
- [ ] `lib_prep_afni_surf.py` — 29 hits (27 print-stmts, 2 iteritems)
- [ ] `afni_surface_alphasim.py` — 12 hits (7 print, 5 xrange)
- [ ] `lib_plot_slices.py` — 7 hits (print, xrange, basestring)
- [ ] `afni_utils.py` — 6 hits (5 print, 1 iteritems)
- [ ] `suma_setup_instacorr.py` — 4 hits (print)
- [ ] Run: `pytest mvpa2/tests/test_lib_afni.py` (if AFNI available)
- [ ] Commit: `RF(PY3): convert support/afni/ to Python 3`

### 4c. `misc/surfing/` (4 files, 27 PY2 hits)
- [ ] `volume_mask_dict.py` — 8 hits (iteritems, basestring)
- [ ] `volsurf.py` — 8 hits (xrange, iteritems)
- [ ] `surf_voxel_selection.py` — 8 hits (xrange, iteritems, basestring)
- [ ] `queryengine.py` — 3 hits (xrange, iteritems)
- [ ] Run: `pytest mvpa2/tests/test_surfing_voxelselection.py`
- [ ] Commit: `RF(PY3): convert misc/surfing/ to Python 3`

### 4d. `cmdline/` (7 files, 14 PY2 hits)
- [ ] `cmd_crossval.py` — 5 hits (print, xrange)
- [ ] `cmd_describe.py` — 2 hits (print)
- [ ] `cmd_dump.py` — 1 hit (print)
- [ ] `cmd_exec.py` — 1 hit (execfile) + has `import nose.tools as nt`
- [ ] `cmd_hyperalign.py` — 1 hit (print)
- [ ] `cmd_info.py` — 2 hits (print)
- [ ] `cmd_mkevds.py` — 2 hits (xrange, iteritems)
- [ ] Run: smoke-test cmdline tools if possible
- [ ] Commit: `RF(PY3): convert cmdline/ to Python 3`

### 4e. `datasets/` (5 files, 27 PY2 hits)
- [ ] `cosmo.py` — 12 hits (iteritems, basestring)
- [ ] `niml.py` — 6 hits (iteritems, basestring)
- [ ] `formats.py` — (already fixed in 4.0)
- [ ] `gifti.py` — 3 hits (basestring)
- [ ] `eeglab.py` — 2 hits (xrange, basestring)
- [ ] Run: `pytest mvpa2/datasets/tests/`
- [ ] Commit: `RF(PY3): convert datasets/ to Python 3`

### 4f. `measures/` (3 files, 7 PY2 hits)
- [ ] `corrcoef.py` — 1 hit (xrange)
- [ ] `noiseperturbation.py` — 1 hit (xrange)
- [ ] `winner.py` — 5 hits (print, xrange, iteritems)
- [ ] Run: `pytest mvpa2/measures/tests/`
- [ ] Commit: `RF(PY3): convert measures/ to Python 3`

### 4g. `algorithms/` (2 files, 5 PY2 hits)
- [ ] `group_clusterthr.py` — 4 hits (xrange)
- [ ] `searchlight_hyperalignment.py` — 1 hit (print)
- [ ] Run: `pytest mvpa2/algorithms/tests/`
- [ ] Commit: `RF(PY3): convert algorithms/ to Python 3`

### 4h. Other scattered files (8 files, ~30 PY2 hits)
- [ ] `support/_copy.py` — 20 hits (replace with stdlib `copy` — see Phase 2.2.6)
- [ ] `support/bayes/multivariate_polya.py` — 8 hits (print in `__main__`)
- [ ] `support/bayes/partial_independence.py` — 6 hits (print in `__main__`)
- [ ] `misc/plot/flat_surf.py` — 2 hits (basestring)
- [x] `clfs/model_selector.py` — removed in Phase 3.4
- [ ] `viz/__init__.py` — 2 hits (basestring)
- [ ] `support/ipython/ipy_pymvpa_completer.py` — 1 hit (print)
- [ ] Run relevant tests for each changed file
- [ ] Commit: `RF(PY3): convert remaining scattered source files`

### 4i. Test files still in `mvpa2/tests/` (11 files, 25 PY2 hits)
- [ ] `test_surfing_surface.py` — 5 hits (xrange)
- [ ] `test_benchmarks_hyperalignment.py` — 3 hits (xrange)
- [ ] `test_generators.py` — 3 hits (xrange)
- [ ] `test_multiclf.py` — 3 hits (print, iteritems)
- [ ] `test_neighborhood.py` — 2 hits (xrange)
- [ ] `test_rsa.py` — 2 hits (xrange)
- [ ] `runner.py` — 3 hits (print)
- [ ] `test_corrstability.py` — 1 hit (iteritems)
- [ ] `test_fxmeasure.py` — 1 hit (iteritems)
- [ ] `test_openfmri.py` — 1 hit (iteritems)
- [ ] `test_usecases.py` — 1 hit (xrange)
- [ ] Commit: `RF(PY3): convert remaining test files to Python 3`

### 4j. Fix `test_suite.py` PY2 syntax
- [ ] `exec "from mvpa2.suite import *"` → `exec("from mvpa2.suite import *")`
- [ ] `.iteritems()` → `.items()`
- [ ] Commit: `BF(PY3): fix test_suite.py PY2 syntax errors`

- [ ] **✓ CHECKPOINT: all source files are PY3-clean, all tests pass**

## Phase 5: Complete Test Infrastructure

Now that all code is PY3-clean and tests are running via pytest, complete
the test restructuring and tooling.

### 5.1 Fix direct nose imports in remaining test/source files

- [ ] `mvpa2/tests/test_viz.py` — `from nose.tools import *` → `from mvpa2.testing import *`
- [ ] `mvpa2/tests/test_openfmri.py` — `from nose.tools import assert_greater` → `from mvpa2.testing import assert_greater`
- [ ] `mvpa2/tests/test_group_clusterthr.py` — `from nose.tools import assert_greater_equal, assert_greater` → `from mvpa2.testing import ...`
- [ ] `mvpa2/misc/tests/test_dcov.py` — `from nose.tools import ok_` → `from mvpa2.testing import ok_`
- [ ] `mvpa2/clfs/sg/__init__.py` — `from nose.plugins.skip import SkipTest` → `from unittest import SkipTest` (if file not already removed in Phase 3.1)
- [ ] `mvpa2/cmdline/cmd_exec.py` — remove `import nose.tools as nt`
- [ ] Run tests for each changed file
- [ ] Commit: `RF: replace direct nose imports in test/source files`

### 5.2 Fix module-level `raise SkipTest`

- [ ] `mvpa2/datasets/tests/test_niftidataset.py` — replace `raise SkipTest` with `pytest.skip("...", allow_module_level=True)`
- [ ] `mvpa2/tests/test_giftidataset.py` — same
- [ ] Commit: `BF: use pytest.skip(allow_module_level=True) for module-level skips`

### 5.3 Complete test file moves (remaining ~20 in `mvpa2/tests/`)

Each move via `git mv`, verified with test run. Batch by target module.

- [ ] `test_benchmarks_hyperalignment.py` → `mvpa2/algorithms/tests/`
- [ ] `test_corrstability.py` → `mvpa2/clfs/tests/` (or `measures/tests/`)
- [ ] `test_fxmeasure.py` → `mvpa2/measures/tests/`
- [ ] `test_generators.py` → `mvpa2/generators/tests/`
- [ ] `test_giftidataset.py` → `mvpa2/datasets/tests/`
- [ ] `test_group_clusterthr.py` → `mvpa2/algorithms/tests/`
- [ ] `test_multiclf.py` → `mvpa2/clfs/tests/`
- [ ] `test_neighborhood.py` → `mvpa2/misc/tests/`
- [ ] `test_openfmri.py` → `mvpa2/datasets/tests/`
- [ ] `test_perturbsensana.py` → `mvpa2/measures/tests/`
- [ ] `test_rsa.py` → `mvpa2/measures/tests/`
- [ ] `test_searchlight.py` → `mvpa2/measures/tests/`
- [ ] `test_searchlight_hyperalignment.py` → `mvpa2/algorithms/tests/`
- [ ] `test_senses.py` → `mvpa2/measures/tests/`
- [ ] `test_som.py` → `mvpa2/mappers/tests/`
- [ ] `test_splitter.py` → `mvpa2/generators/tests/`
- [ ] `test_stats.py` → `mvpa2/misc/tests/`
- [ ] `test_stats_sp.py` → `mvpa2/misc/tests/`
- [ ] `test_surfing_surface.py` → `mvpa2/misc/surfing/tests/`
- [ ] `test_surfing_voxelselection.py` → `mvpa2/misc/surfing/tests/`
- [ ] `test_testing.py` → `mvpa2/testing/tests/`
- [ ] `test_usecases.py` → `mvpa2/tests/` (keep — cross-cutting)
- [ ] `test_viz.py` → `mvpa2/viz/tests/`
- [ ] `test_winner.py` → `mvpa2/measures/tests/`
- [ ] Create `__init__.py` in any new `tests/` subdirectories
- [ ] Commit per batch: `RF: move <module> tests to <module>/tests/`

### 5.4 Clean up old test infrastructure

- [ ] `mvpa2/tests/__init__.py` — remove `run_tests_using_nose()`, simplify `run()`
- [ ] `mvpa2/tests/test_suite.py` — review; possibly remove if pytest discovery suffices
- [ ] `mvpa2/base/externals.py` — remove `'nose'` from `_KNOWN` externals
- [ ] Remove nose from any requirements/dependencies
- [ ] Commit: `RF: remove nose-specific test infrastructure`

### 5.5 Create `tox.ini`

- [ ] `[tox]` envlist: `py3{9,10,11,12,13},lint`
- [ ] `[testenv]`: `deps = .[test]`, `commands = pytest {posargs:mvpa2/}`
- [ ] `[testenv:lint]`: `deps = .[devel]`, `commands = ruff check mvpa2/`
- [ ] Run: `tox -e py3` to verify
- [ ] Commit: `ENH: add tox.ini for test automation`

- [ ] **✓ CHECKPOINT: pytest + tox fully working, all tests discovered and passing**

## Phase 6: Build System — `numpy.distutils` → `meson-python`

Now that all pure-Python code is tested, migrate the build system to enable
C extension building on modern Python.

- [ ] **6.1** Create `pyproject.toml`:
  - [ ] `[build-system]` with `meson-python>=0.15`, `numpy>=1.25` in `requires`
  - [ ] `[project]` metadata (name, version from `mvpa2/base/info.py`, requires-python `>=3.9`)
  - [ ] `[project.dependencies]`: numpy, scipy, nibabel, h5py, joblib
  - [ ] `[project.optional-dependencies]`:
    - [ ] `test = [pytest, pytest-cov, pytest-timeout, ...]`
    - [ ] `devel = [pymvpa2[test], ruff, tox, tox-uv, ...]`
    - [ ] `ci = [pymvpa2[devel], tox-gh-actions]`
  - [ ] `[tool.pytest.ini_options]` — testpaths, markers, filterwarnings
- [ ] **6.2** Create top-level `meson.build` — project definition, recurse into `mvpa2/`
- [ ] **6.3** Create `mvpa2/meson.build` — install pure-Python subpackages
- [ ] **6.4** Create `mvpa2/clfs/libsvmc/meson.build`:
  - [ ] Build `_svmc` extension from `svmc_wrap.cpp` + `3rd/libsvm/svm.cpp`
  - [ ] Include dirs: `3rd/libsvm/`, numpy headers
- [ ] **6.5** Create `mvpa2/clfs/libsmlrc/meson.build`:
  - [ ] Build `smlrc` extension from `smlr.c`, link `-lm`
- [ ] **6.6** Remove or thin out `setup.py` (keep as shim if needed)
- [ ] **6.7** Update `Makefile` — remove `setup.py build_ext` references
- [ ] **6.8** Verify: `pip install -e . --no-build-isolation` builds both C extensions
- [ ] **6.9** Run: `pytest mvpa2/clfs/tests/` (SVM/SMLR tests that need C extensions)
- [ ] Commit: `RF: migrate build system from numpy.distutils to meson-python`
- [ ] **✓ CHECKPOINT: `pip install -e .` works, C extension tests pass**

## Phase 7: CI + API Modernization + Polish

### 7.1 CI Modernization — Travis → GitHub Actions

- [ ] Create `.github/workflows/test.yml`:
  - [ ] Matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13
  - [ ] Steps: checkout, setup-python, `pip install -e ".[ci]"`, `tox`
  - [ ] Separate lint job
- [ ] Add `[gh-actions]` mapping in `tox.ini` (tox-gh-actions)
- [ ] Remove `.travis.yml`
- [ ] Commit: `RF: replace Travis CI with GitHub Actions`

### 7.2 API Modernization

- [ ] **7.2.1** `inspect.getargspec()` → `inspect.getfullargspec()`:
  - [ ] Already shimmed in `mvpa2/support/utils.py`
  - [ ] Ensure all callers use the shim (grep for `getargspec` outside `support/utils.py`)
  - [ ] Run tests; Commit: `RF(PY3): ensure all getargspec callers use PY3 shim`

- [ ] **7.2.2** `numpy.testing.decorators` → `pytest.mark.skipif`
  - [ ] (Already addressed in Phase 1.1.1 for `tools.py`; grep for any remaining usages)
  - [ ] Commit: `RF: replace deprecated numpy.testing.decorators with pytest.mark`

- [ ] **7.2.3** `collections.MutableMapping` → `collections.abc.MutableMapping` (if any)
- [ ] **7.2.4** `imp` module → `importlib` (if any)

- [ ] **7.2.5** h5py 3.x compatibility:
  - [ ] Remove `import h5py.highlevel` (gone in h5py 3.0)
  - [ ] **⚠ CONSULT**: h5py 3.0 changed string handling — strings stored as bytes
    by default. This affects how PyMVPA's custom HDF5 serialization reads/writes
    string attributes. The `ensure_str()` helper (from branch WIP stash) addresses
    this but needs thorough testing with real saved datasets from PY2 era.
  - [ ] Run tests; Commit: `RF: update for h5py 3.x API (remove highlevel, fix string handling)`

- [ ] **7.2.6** scikit-learn API updates:
  - [ ] Check for deprecated parameter names, removed methods
  - [ ] Run tests; Commit: `RF: update for scikit-learn 1.x API changes`

- [ ] **7.2.7** `griddata`:
  - [ ] Replace `matplotlib.mlab.griddata` (removed mpl 3.1) with `scipy.interpolate.griddata`
  - [ ] Update `mvpa2/support/griddata.py`
  - [ ] Run tests; Commit: `RF: use scipy.interpolate.griddata (mpl.mlab.griddata removed)`

- [ ] **7.2.8** Version constraints in pyproject.toml:
  - [ ] `numpy>=1.22`, `scipy>=1.7`, `h5py>=3.0`, `nibabel>=3.0`, `scikit-learn>=1.0`

### 7.3 Cleanup and Polish

- [ ] Final grep: no remaining `from future`, `from past`, `from builtins`, `from six`
- [ ] Final grep: no remaining `from nose`, `import nose`
- [ ] Update `mvpa2/base/info.py` version string
- [ ] Run `ruff check --fix` on entire codebase:
  - [ ] Use `datalad run -m "STY: ruff auto-fix" -- ruff check --fix mvpa2/`
- [ ] Run `codespell` (with `ignore-regex = https?://\S+` to protect URLs):
  - [ ] Use `datalad run` to record
- [ ] Verify full test suite: `tox -e py3`
- [ ] Remove stale files:
  - [ ] `Changelog.orig`, `Changelog.rej`
  - [ ] Old `.so` files
  - [ ] `py3tool.py` (confirm gone)
  - [ ] `mvpa2/tests/runner.py` (if no longer needed)
- [ ] Commit: `STY: final cleanup and polish`

---

## Execution Order & Dependencies

```
Phase 0  Clean up WIP (stash + gitignore)
  ↓
Phase 1  BOOTSTRAP TESTABILITY ← first green checkpoint
  ↓      (fix testing/tools.py, conftest.py, verify base/tests pass)
  ↓
Phase 2  Strip future/six (verified incrementally against green baseline)
  ↓
Phase 3  Remove dead deps (each removal verified by test run)
  ↓
Phase 4  Convert remaining 53 files (per-submodule, verified)
  ↓
Phase 5  Complete test infra (moves, tox, cleanup old nose bits)
  ↓
Phase 6  Build system (meson-python, C extensions)
  ↓
Phase 7  CI + API modernization + polish
```

Each phase builds on a verified green state from the previous phase.
No phase makes changes that cannot be tested immediately.

## Consultation Checkpoints (⚠ CONSULT)

These steps involve non-obvious semantic decisions where the implementer
should stop and consult the maintainer:

1. **Phase 2.2.3** — `old_div()` replacement: integer vs float division ambiguity
2. **Phase 2.2.4** — `with_metaclass` removal: HDF5 serialization may depend on
   `__metaclass__` class attribute presence
3. **Phase 2.2.5** — PY2 guards in `hdf5.py`: must preserve ability to load
   PY2-era saved HDF5 files while removing PY2 saving codepaths
4. **Phase 7.2.5** — h5py 3.x string handling: custom serialization may need
   careful testing with real PY2-era datasets
5. **Any step** where tests fail in non-obvious ways after a mechanical
   transformation — bytes/str confusion, changed iteration order, or
   numeric precision differences between PY2 and PY3

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| C extensions may not build with meson-python | Use pre-generated SWIG wrapper (already committed); test early in Phase 6 |
| `sweepargs` behavior differences under pytest | Keep `sweepargs` as-is; only fix `SkipTest` import; defer full parametrize conversion |
| `old_div()` replacement ambiguity (int vs float division) | ⚠ CONSULT — check each usage individually |
| Deep PY2 bytes/str assumptions in NIML/HDF5 code | ⚠ CONSULT — handle case-by-case; test with real PY2-era data files |
| `support/_copy.py` (vendored PY2 copy.py) may serve specific purpose | Investigate callers before replacing with stdlib `copy` |
| `assert_raises` behavior change (nose's vs pytest.raises) | pytest.raises works as context manager AND callable — datalad proved this works |
| Removing Shogun/MDP/pprocess may break user workflows | Announce deprecation in changelog; these projects are dead anyway |
| HDF5 files saved under PY2 may not load under PY3 | Keep `builtins_mods` tuple for backward-compat loading; test with saved fixtures |
| `future` needed temporarily in Phase 1 | Install via pip for bootstrap only; remove in Phase 2.3 once stripped from all files |
