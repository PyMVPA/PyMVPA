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

## Phase 0: Clean Up WIP and Establish Baseline

- [ ] **0.1** Stash all uncommitted working tree changes for reference:
  ```
  git stash push -m "WIP: in-progress py3 fixes (collections, hdf5, transerror, etc.)" \
    mvpa2/base/collections.py mvpa2/base/hdf5.py mvpa2/base/tests/test_hdf5.py \
    mvpa2/clfs/transerror.py mvpa2/datasets/sources/skl_data.py \
    mvpa2/misc/support.py mvpa2/support/utils.py mvpa2/tests/test_transerror.py
  ```
  These can be cherry-picked or referenced later when re-approaching each
  submodule. The stash preserves the prior work without polluting the
  commit history with debug artifacts (pdb.set_trace, etc.).
- [ ] **0.2** Confirm `py3tool.py` is deleted (already removed on branch)
- [ ] **0.3** Remove pre-built `.so` files for old Python versions from `mvpa2/clfs/lib*/`
- [ ] **0.4** Add/update `.gitignore`:
  - [ ] `*.so`, `*.pyc`, `__pycache__/`
  - [ ] `*.egg-info/`, `.eggs/`
  - [ ] `.cache/`, `.hypothesis/`, `.pytest_cache/`
  - [ ] `.tox/`, `.venv/`
  - [ ] `.npm/`
- [ ] **0.5** Commit: `RF: clean up .gitignore and remove stale build artifacts`

## Phase 1: Bootstrap Testability (THE CRITICAL FIRST MILESTONE)

Goal: get `PYTHONPATH=. pytest mvpa2/base/tests/` green on Python 3.11+.
Keep `future`/`six` installed temporarily — they are needed by the
already-converted source files until Phase 2 strips them.

The only files that change here are in `mvpa2/testing/` — no test file
changes, no source module changes.

### 1.1 Fix immediate crashers in `mvpa2/testing/tools.py`

- [ ] **1.1.1** Fix `numpy.testing.dec`/`numpy.testing.decorators` crash:
  - [ ] Replace the try/except chain with `pytest.mark.skipif` (or a
    local stub that delegates to `pytest.mark.skipif`)
  - [ ] This is the #1 blocker — without this fix, `from mvpa2.testing import *`
    crashes on any modern numpy
- [ ] **1.1.2** Fix `from six import reraise as raise_`:
  - [ ] Replace with native: `raise exc.with_traceback(tb)` inline at usage site
  - [ ] (Used in `labile` decorator)
- [ ] **1.1.3** Replace nose assertion imports with thin shims
  (modeled on datalad's `utils_pytest.py`):
  - [ ] Remove entire `if externals.exists('nose'):` / `else:` block
  - [ ] Add `from unittest import SkipTest`
  - [ ] Add thin assertion shims:
    ```python
    import pytest

    def ok_(val, msg=None):
        assert val, msg

    def eq_(a, b, msg=None):
        assert a == b, msg or f"{a!r} != {b!r}"

    def assert_true(val, msg=None):
        assert val, msg

    def assert_false(val, msg=None):
        assert not val, msg

    def assert_equal(first, second, msg=None):
        assert first == second, msg or f"{first!r} != {second!r}"

    assert_equals = assert_equal

    def assert_not_equal(first, second, msg=None):
        assert first != second, msg

    assert_not_equals = assert_not_equal

    def assert_in(first, second, msg=None):
        assert first in second, msg

    def assert_not_in(first, second, msg=None):
        assert first not in second, msg

    def assert_is_instance(obj, cls, msg=None):
        assert isinstance(obj, cls), msg

    def assert_greater(first, second, msg=None):
        assert first > second, msg

    def assert_greater_equal(first, second, msg=None):
        assert first >= second, msg

    assert_raises = pytest.raises
    ```
- [ ] **1.1.4** Replace all 4 `make_decorator` uses with `functools.wraps`:
  - [ ] `with_tempfile`
  - [ ] `reseed_rng`
  - [ ] `nodebug`
  - [ ] `labile`
- [ ] **1.1.5** Drop unused nose-only imports: `timed`, `with_setup`, `raises`,
  `istest`, `nottest`

### 1.2 Fix `sweepargs`

- [ ] **1.2.1** `mvpa2/testing/sweep.py` — verify `SkipTest` now comes from
  `unittest` via `mvpa2.testing.tools`. `except SkipTest` works identically.

### 1.3 Create minimal `conftest.py`

- [ ] **1.3.1** Root `conftest.py`:
  - [ ] Session-scoped autouse fixture wrapping `mvpa2.setup_module()` /
    `teardown_module()` logic (debug targets, RNG seed, warning suppression)
  - [ ] `collect_ignore` list for `mvpa2/tests/__init__.py`, `mvpa2/tests/runner.py`
  - [ ] Register custom markers: `labile`, `slow`, `ai_generated`

### 1.4 Verify

- [ ] **1.4.1** Install temporary deps: `pip install future six`
- [ ] **1.4.2** Run: `PYTHONPATH=. pytest mvpa2/base/tests/ -v`
- [ ] **1.4.3** Identify and fix any remaining failures
- [ ] **✓ CHECKPOINT: `mvpa2/base/tests/` passes on Python 3.11+**
- [ ] Commit: `RF: replace nose with pytest-compatible shims in testing infrastructure`
- [ ] Extend to other already-converted test directories:
  - [ ] `pytest mvpa2/algorithms/tests/`
  - [ ] `pytest mvpa2/atlases/tests/`
  - [ ] `pytest mvpa2/clfs/tests/`
  - [ ] `pytest mvpa2/datasets/tests/`
  - [ ] `pytest mvpa2/featsel/tests/`
  - [ ] `pytest mvpa2/kernels/tests/`
  - [ ] `pytest mvpa2/mappers/tests/`
  - [ ] `pytest mvpa2/measures/tests/`
  - [ ] `pytest mvpa2/misc/tests/`
  - [ ] `pytest mvpa2/support/tests/`
  - [ ] Fix any failures found, commit each fix separately
- [ ] **✓ CHECKPOINT: all already-moved tests pass**

## Phase 2: Strip `future`/`six`/`past` — Pure Python 3

Now that we have a green baseline, strip the compatibility layer incrementally.
After each step, re-run the passing test suites to confirm no regressions.

Use `datalad run` for each scripted bulk transformation to record what was run.

### 2.1 Scripted bulk removals (safe — these imports are no-ops in PY3)

- [ ] **2.1.1** Remove `from __future__ import` lines across all files:
  - [ ] Use `datalad run -m "RF(PY3): strip __future__ imports" -- <script>`
  - [ ] `division`, `print_function`, `unicode_literals`, `absolute_import`
- [ ] **2.1.2** Remove `from builtins import` lines:
  - [ ] Use `datalad run -m "RF(PY3): strip builtins imports" -- <script>`
  - [ ] `str`, `range`, `zip`, `map`, `filter`, `next`, `object`, `bytes`
- [ ] **2.1.3** Remove `from future import standard_library; standard_library.install_aliases()`:
  - [ ] Use `datalad run -m "RF(PY3): strip standard_library.install_aliases" -- <script>`
- [ ] Run all tests after each step
- [ ] **✓ CHECKPOINT: all tests still pass without `from __future__`/`builtins`/`standard_library`**

### 2.2 Replacements requiring per-file review

- [ ] **2.2.1** Remove `from past.builtins import basestring`:
  - [ ] Replace all `basestring` usage with `str`
  - [ ] Use `datalad run` for the bulk sed, then manually verify edge cases
  - [ ] Run tests
  - [ ] Commit: `RF(PY3): replace basestring with str`

- [ ] **2.2.2** Remove `from past.builtins import cmp`:
  - [ ] Provide `cmp` as a utility: `cmp = lambda a, b: (a > b) - (a < b)` in `mvpa2/misc/support.py`
  - [ ] Update callers to import from there, or inline
  - [ ] Run tests
  - [ ] Commit: `RF(PY3): provide native cmp() replacement`

- [ ] **2.2.3** Remove `from past.utils import old_div`:
  - [ ] **⚠ CONSULT**: Review each `old_div(a, b)` usage to determine correct
    replacement (`a // b` for integer division, `a / b` for float). The `old_div`
    function replicates PY2's `/` operator semantics which does integer division
    when both operands are ints. Mistakes here silently produce wrong results.
  - [ ] Affected files: `mvpa2/atlases/base.py`, `mvpa2/atlases/transformation.py`,
    `mvpa2/misc/support.py`, `mvpa2/misc/io/base.py`, and others
  - [ ] Commit per submodule after verifying tests pass

- [ ] **2.2.4** Remove `from future.utils import with_metaclass`:
  - [ ] Replace `class Foo(with_metaclass(Meta, Base)):` → `class Foo(Base, metaclass=Meta):`
  - [ ] **⚠ CONSULT**: The `__metaclass__` attribute was deliberately kept in some
    places for HDF5 serialization compatibility. Check if `hdf5.py` depends on
    `__metaclass__` class attribute being present for reconstructing saved objects.
  - [ ] Run tests (especially `mvpa2/base/tests/test_hdf5.py`)
  - [ ] Commit: `RF(PY3): use native metaclass syntax`

- [ ] **2.2.5** Remove PY2 guards:
  - [ ] `PY2 = sys.version_info[0] == 2` in `mvpa2/misc/support.py` and all `if PY2:` branches
  - [ ] **⚠ CONSULT**: PY2-specific codepaths in `mvpa2/base/hdf5.py` — this is the
    most delicate area. The `builtins_mods = ('__builtin__', 'builtins')` tuple
    must be kept so that HDF5 files saved under PY2 can still be loaded under PY3
    (backward compatibility for stored data). Only the PY2 *saving* codepath can
    be removed; the PY2-format *loading* codepath must remain.
  - [ ] Run tests
  - [ ] Commit: `RF(PY3): remove PY2 runtime guards (keep PY2-data loading compat)`

- [ ] **2.2.6** Remove `mvpa2/support/_copy.py`:
  - [ ] Find all imports: `from mvpa2.support._copy import deepcopy` → `from copy import deepcopy`
  - [ ] Run tests
  - [ ] Commit: `RF(PY3): use stdlib copy instead of vendored PY2 copy`

### 2.3 Final future/six cleanup

- [ ] **2.3.1** Remove `future` and `six` from dependencies/externals:
  - [ ] `mvpa2/base/externals.py` — remove `'future'`, `'six'` entries
  - [ ] Remove `cPickle` from `_KNOWN` externals (it's `pickle` in PY3)
  - [ ] Commit: `RF(PY3): remove future/six/cPickle from externals`
- [ ] **2.3.2** Clean up `SyntaxError` suppression:
  - [ ] `mvpa2/base/externals.py` — remove `'SyntaxError'` from caught exceptions
    (was added as TODO during 2to3)
  - [ ] Commit: `RF(PY3): remove SyntaxError suppression in externals checking`
- [ ] **2.3.3** Uninstall `future` and `six` from dev environment
- [ ] Run all tests
- [ ] **✓ CHECKPOINT: all tests pass WITHOUT `future`/`six` installed**

## Phase 3: Remove Dead Dependencies

Each removal is surgical and independently verifiable. Run the full test
suite after each removal to catch any unexpected coupling.

- [ ] **3.1** Remove Shogun support entirely:
  - [ ] `mvpa2/clfs/sg/` — entire directory (svm.py, sens.py, __init__.py)
  - [ ] Remove all `shogun`/`sg*` entries from `mvpa2/base/externals.py`
  - [ ] Remove shogun entries from `mvpa2/clfs/warehouse.py`
  - [ ] Remove shogun from `mvpa2/testing/clfs.py`
  - [ ] Remove `mvpa2/kernels/sg.py`
  - [ ] Remove related tests: `test_svmkernels.py` shogun parts, `test_kernel.py` shogun parts
  - [ ] Run: `pytest mvpa2/clfs/tests/ mvpa2/kernels/tests/`
  - [ ] Commit: `RF: remove Shogun support (project unmaintained since 2019)`

- [ ] **3.2** Remove MDP support:
  - [ ] `mvpa2/mappers/mdp_adaptor.py` — remove
  - [ ] `mvpa2/mappers/lle.py` — remove (MDP-based LLE; sklearn has LLE)
  - [ ] Remove `mdp`/`mdp ge 2.4` entries from externals
  - [ ] Remove related tests
  - [ ] Run: `pytest mvpa2/mappers/tests/`
  - [ ] Commit: `RF: remove MDP support (project unmaintained since 2020)`

- [ ] **3.3** Remove pprocess support:
  - [ ] Remove pprocess codepaths in `mvpa2/measures/searchlight.py`,
    `misc/surfing/surf_voxel_selection.py`, `algorithms/searchlight_hyperalignment.py`
  - [ ] Remove `pprocess` from externals
  - [ ] Run: `pytest mvpa2/measures/tests/ mvpa2/algorithms/tests/`
  - [ ] Commit: `RF: remove pprocess support (use joblib instead)`

- [ ] **3.4** Remove OpenOpt support:
  - [ ] Remove openopt codepaths in `mvpa2/clfs/gpr.py`, `model_selector.py`
  - [ ] Remove `openopt` from externals
  - [ ] Run: `pytest mvpa2/clfs/tests/`
  - [ ] Commit: `RF: remove OpenOpt support (project dead since ~2014)`

- [ ] **3.5** Remove scipy.weave/weave support:
  - [ ] Remove weave codepath in `mvpa2/clfs/distance.py`
  - [ ] Remove `weave`/`scipy.weave` from externals
  - [ ] Run: `pytest mvpa2/clfs/tests/`
  - [ ] Commit: `RF: remove scipy.weave support (removed from scipy in 1.0)`

- [ ] **3.6** Clean up remaining dead externals:
  - [ ] Remove `cPickle`, `hcluster`, `nose`, `mock` (use `unittest.mock`) from externals
  - [ ] Run all tests
  - [ ] Commit: `RF: remove dead external checks`

- [ ] **✓ CHECKPOINT: full test suite passes, dead code removed**

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
- [ ] `clfs/model_selector.py` — 2 hits (print)
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
