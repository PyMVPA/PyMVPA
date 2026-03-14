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

### Minimal diff for reviewability
The primary non-functional goal is keeping diffs small and reviewable.
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
| **nose** | 1.3.7 (Jun 2015) | Broken ≥3.10 | Replace with pytest shims (Phase 4) |
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

## Phase 1: Build System — `numpy.distutils` → `meson-python`

- [ ] **1.1** Create `pyproject.toml`:
  - [ ] `[build-system]` with `meson-python>=0.15`, `numpy>=1.25` in `requires`
  - [ ] `[project]` metadata (name, version from `mvpa2/base/info.py`, requires-python `>=3.9`)
  - [ ] `[project.dependencies]`: numpy, scipy, nibabel, h5py, joblib
  - [ ] `[project.optional-dependencies]`:
    - [ ] `test = [pytest, pytest-cov, pytest-timeout, ...]`
    - [ ] `devel = [pymvpa2[test], ruff, tox, tox-uv, ...]`
    - [ ] `ci = [pymvpa2[devel], tox-gh-actions]`
  - [ ] `[tool.pytest.ini_options]` — testpaths, markers, filterwarnings
- [ ] **1.2** Create top-level `meson.build` — project definition, recurse into `mvpa2/`
- [ ] **1.3** Create `mvpa2/meson.build` — install pure-Python subpackages
- [ ] **1.4** Create `mvpa2/clfs/libsvmc/meson.build`:
  - [ ] Build `_svmc` extension from `svmc_wrap.cpp` + `3rd/libsvm/svm.cpp`
  - [ ] Include dirs: `3rd/libsvm/`, numpy headers
- [ ] **1.5** Create `mvpa2/clfs/libsmlrc/meson.build`:
  - [ ] Build `smlrc` extension from `smlr.c`, link `-lm`
- [ ] **1.6** Remove or thin out `setup.py` (keep as shim if needed)
- [ ] **1.7** Update `Makefile` — remove `setup.py build_ext` references
- [ ] **1.8** Verify: `pip install -e . --no-build-isolation` builds both C extensions
- [ ] **1.9** Commit: `RF: migrate build system from numpy.distutils to meson-python`

## Phase 2: Strip `future`/`six`/`past` Library — Pure Python 3

Use `datalad run` for each scripted bulk transformation to record what was run.

- [ ] **2.1** Script: remove `from __future__ import` lines across all files
  - [ ] Use `datalad run -m "RF(PY3): strip __future__ imports" -- <sed/python script>`
  - [ ] `division`, `print_function`, `unicode_literals`, `absolute_import`
- [ ] **2.2** Script: remove `from builtins import` lines
  - [ ] Use `datalad run -m "RF(PY3): strip builtins imports" -- <script>`
  - [ ] `str`, `range`, `zip`, `map`, `filter`, `next`, `object`, `bytes`
- [ ] **2.3** Script: remove `from future import standard_library; standard_library.install_aliases()`
  - [ ] Use `datalad run -m "RF(PY3): strip standard_library.install_aliases" -- <script>`
- [ ] **2.4** Remove `from past.builtins import basestring`:
  - [ ] Replace all `basestring` usage with `str`
  - [ ] Files: `mvpa2/base/collections.py`, `mvpa2/base/dataset.py`, `mvpa2/base/dochelpers.py`, `mvpa2/base/state.py`, `mvpa2/clfs/meta.py`, `mvpa2/mappers/base.py`, `mvpa2/misc/support.py`, and ~20 others
  - [ ] Use `datalad run` for the bulk sed, then manually verify edge cases
  - [ ] Commit: `RF(PY3): replace basestring with str`
- [ ] **2.5** Remove `from past.builtins import cmp`:
  - [ ] Provide `cmp` as a utility: `cmp = lambda a, b: (a > b) - (a < b)` in `mvpa2/misc/support.py`
  - [ ] Update callers to import from there, or inline
  - [ ] Commit: `RF(PY3): provide native cmp() replacement`
- [ ] **2.6** Remove `from past.utils import old_div`:
  - [ ] **⚠ CONSULT**: Review each `old_div(a, b)` usage to determine correct
    replacement (`a // b` for integer division, `a / b` for float). The `old_div`
    function replicates PY2's `/` operator semantics which does integer division
    when both operands are ints. Mistakes here silently produce wrong results.
  - [ ] Affected files: `mvpa2/atlases/base.py`, `mvpa2/atlases/transformation.py`, `mvpa2/misc/support.py`, `mvpa2/misc/io/base.py`, and others
  - [ ] Commit per submodule after verifying tests pass
- [ ] **2.7** Remove `from future.utils import with_metaclass`:
  - [ ] Replace `class Foo(with_metaclass(Meta, Base)):` → `class Foo(Base, metaclass=Meta):`
  - [ ] **⚠ CONSULT**: The `__metaclass__` attribute was deliberately kept in some
    places for HDF5 serialization compatibility. Check if `hdf5.py` depends on
    `__metaclass__` class attribute being present for reconstructing saved objects.
  - [ ] Commit: `RF(PY3): use native metaclass syntax`
- [ ] **2.8** Remove `from six import reraise`:
  - [ ] Replace with native `raise exc.with_traceback(tb)` in `mvpa2/testing/sweep.py`
  - [ ] Commit: `RF(PY3): replace six.reraise with native traceback syntax`
- [ ] **2.9** Remove PY2 guards:
  - [ ] `PY2 = sys.version_info[0] == 2` in `mvpa2/misc/support.py` and all `if PY2:` branches
  - [ ] **⚠ CONSULT**: PY2-specific codepaths in `mvpa2/base/hdf5.py` — this is the
    most delicate area. The HDF5 module has custom serialization for various types
    (including `__reduce__` protocol, builtins module name handling, bytes/str
    decoding of stored attributes). The `builtins_mods = ('__builtin__', 'builtins')`
    tuple must be kept so that HDF5 files saved under PY2 can still be loaded under
    PY3 (backward compatibility for stored data). Only the PY2 *saving* codepath
    can be removed; the PY2-format *loading* codepath must remain.
  - [ ] Commit: `RF(PY3): remove PY2 runtime guards (keep PY2-data loading compat)`
- [ ] **2.10** Remove `mvpa2/support/_copy.py`:
  - [ ] Find all imports: `from mvpa2.support._copy import deepcopy` → `from copy import deepcopy`
  - [ ] Commit: `RF(PY3): use stdlib copy instead of vendored PY2 copy`
- [ ] **2.11** Remove `future` and `six` from dependencies/externals:
  - [ ] `mvpa2/base/externals.py` — remove `'future'`, `'six'` entries
  - [ ] Remove `cPickle` from `_KNOWN` externals (it's `pickle` in PY3)
  - [ ] Commit: `RF(PY3): remove future/six/cPickle from externals`
- [ ] **2.12** Clean up `SyntaxError` suppression:
  - [ ] `mvpa2/base/externals.py` — remove `'SyntaxError'` from caught exceptions (was added as TODO during 2to3)
  - [ ] Commit: `RF(PY3): remove SyntaxError suppression in externals checking`

## Phase 3: Convert Remaining ~53 Unconverted Source Files

All mechanical PY3 fixes. Patterns:
`xrange→range`, `.iteritems()→.items()`, `.itervalues()→.values()`,
`.iterkeys()→.keys()`, `print x→print(x)`, `raise X, msg→raise X(msg)`,
`except X, e→except X as e`, `basestring→str`, `execfile(f)→exec(open(f).read())`,
`has_key(k)→k in d`

**Commit discipline**: One commit per submodule group (3a, 3b, etc.).
Run related tests after each conversion. Use `datalad run` for any scripted
transformations.

### 3a. `support/nibabel/` (8 files, 43 PY2 hits)
- [ ] `afni_niml.py` — 14 hits (xrange, iteritems, basestring)
- [ ] `afni_niml_dset.py` — 10 hits (xrange, iteritems, basestring)
- [ ] `afni_suma_spec.py` — 7 hits (print, xrange, iteritems, basestring)
- [ ] `afni_niml_annot.py` — 5 hits (xrange, basestring)
- [ ] `afni_suma_1d.py` — 2 hits (xrange, basestring)
- [ ] `afni_niml_roi.py` — 1 hit (basestring)
- [ ] `surf_fs_asc.py` — 2 hits (xrange)
- [ ] `surf_gifti.py` — 2 hits (iteritems)
- [ ] Run: `pytest mvpa2/tests/test_surfing*.py` (or wherever surfing tests live)
- [ ] Commit: `RF(PY3): convert support/nibabel/ to Python 3`

### 3b. `support/afni/` (5 files, 58 PY2 hits)
- [ ] `lib_prep_afni_surf.py` — 29 hits (27 print-stmts, 2 iteritems)
- [ ] `afni_surface_alphasim.py` — 12 hits (7 print, 5 xrange)
- [ ] `lib_plot_slices.py` — 7 hits (print, xrange, basestring)
- [ ] `afni_utils.py` — 6 hits (5 print, 1 iteritems)
- [ ] `suma_setup_instacorr.py` — 4 hits (print)
- [ ] Run: `pytest mvpa2/tests/test_lib_afni.py` (if AFNI available)
- [ ] Commit: `RF(PY3): convert support/afni/ to Python 3`

### 3c. `misc/surfing/` (4 files, 27 PY2 hits)
- [ ] `volume_mask_dict.py` — 8 hits (iteritems, basestring)
- [ ] `volsurf.py` — 8 hits (xrange, iteritems)
- [ ] `surf_voxel_selection.py` — 8 hits (xrange, iteritems, basestring)
- [ ] `queryengine.py` — 3 hits (xrange, iteritems)
- [ ] Run: `pytest mvpa2/tests/test_surfing_voxelselection.py`
- [ ] Commit: `RF(PY3): convert misc/surfing/ to Python 3`

### 3d. `cmdline/` (7 files, 14 PY2 hits)
- [ ] `cmd_crossval.py` — 5 hits (print, xrange)
- [ ] `cmd_describe.py` — 2 hits (print)
- [ ] `cmd_dump.py` — 1 hit (print)
- [ ] `cmd_exec.py` — 1 hit (execfile) + has `import nose.tools as nt`
- [ ] `cmd_hyperalign.py` — 1 hit (print)
- [ ] `cmd_info.py` — 2 hits (print)
- [ ] `cmd_mkevds.py` — 2 hits (xrange, iteritems)
- [ ] Run: smoke-test cmdline tools if possible
- [ ] Commit: `RF(PY3): convert cmdline/ to Python 3`

### 3e. `datasets/` (5 files, 27 PY2 hits)
- [ ] `cosmo.py` — 12 hits (iteritems, basestring)
- [ ] `niml.py` — 6 hits (iteritems, basestring)
- [ ] `formats.py` — 4 hits (raise-comma — **syntax error in PY3, must fix first**)
- [ ] `gifti.py` — 3 hits (basestring)
- [ ] `eeglab.py` — 2 hits (xrange, basestring)
- [ ] Run: `pytest mvpa2/datasets/tests/`
- [ ] Commit: `RF(PY3): convert datasets/ to Python 3`

### 3f. `measures/` (3 files, 7 PY2 hits)
- [ ] `corrcoef.py` — 1 hit (xrange)
- [ ] `noiseperturbation.py` — 1 hit (xrange)
- [ ] `winner.py` — 5 hits (print, xrange, iteritems)
- [ ] Run: `pytest mvpa2/measures/tests/` and `pytest mvpa2/tests/test_winner.py`
- [ ] Commit: `RF(PY3): convert measures/ to Python 3`

### 3g. `algorithms/` (2 files, 5 PY2 hits)
- [ ] `group_clusterthr.py` — 4 hits (xrange)
- [ ] `searchlight_hyperalignment.py` — 1 hit (print)
- [ ] Run: `pytest mvpa2/algorithms/tests/`
- [ ] Commit: `RF(PY3): convert algorithms/ to Python 3`

### 3h. Other scattered files (8 files, ~30 PY2 hits)
- [ ] `support/_copy.py` — 20 hits (replace with stdlib `copy` — see Phase 2.10)
- [ ] `support/bayes/multivariate_polya.py` — 8 hits (print in `__main__`)
- [ ] `support/bayes/partial_independence.py` — 6 hits (print in `__main__`)
- [ ] `misc/plot/flat_surf.py` — 2 hits (basestring)
- [ ] `mappers/som.py` — 2 hits (xrange, raise-comma — **syntax error in PY3**)
- [ ] `clfs/model_selector.py` — 2 hits (print)
- [ ] `viz/__init__.py` — 2 hits (basestring)
- [ ] `support/ipython/ipy_pymvpa_completer.py` — 1 hit (print)
- [ ] Run relevant tests for each changed file
- [ ] Commit: `RF(PY3): convert remaining scattered source files`

### 3i. Test files still in `mvpa2/tests/` (11 files, 25 PY2 hits)
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

### 3j. Fix `test_suite.py` PY2 syntax
- [ ] `exec "from mvpa2.suite import *"` → `exec("from mvpa2.suite import *")`
- [ ] `.iteritems()` → `.items()`
- [ ] Commit: `BF(PY3): fix test_suite.py PY2 syntax errors`

## Phase 4: Test Infrastructure — Nose → Pytest (Minimal-Diff)

### 4.1 Create compatibility shims in `mvpa2/testing/tools.py`

Replace the `if externals.exists('nose'):` block with thin wrappers
(modeled on datalad's `utils_pytest.py`):

- [ ] **4.1.1** Remove the entire `if externals.exists('nose'):` / `else:` block
- [ ] **4.1.2** Add `from unittest import SkipTest` (pytest recognizes this)
- [ ] **4.1.3** Add thin assertion shims (no test file changes needed):
  ```python
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
- [ ] **4.1.4** Replace all 4 `make_decorator` uses with `functools.wraps`:
  - [ ] `with_tempfile` (line ~329)
  - [ ] `reseed_rng` (line ~361)
  - [ ] `nodebug` (line ~404)
  - [ ] `labile` (line ~463)
- [ ] **4.1.5** Replace `skipif` from `numpy.testing.dec` with `pytest.mark.skipif`
- [ ] **4.1.6** Drop unused imports: `timed`, `with_setup`, `raises`, `istest`, `nottest`
- [ ] **4.1.7** Run: `pytest mvpa2/base/tests/` to verify shims work
- [ ] Commit: `RF: replace nose imports with pytest-compatible shims in testing/tools.py`

### 4.2 Fix `sweepargs` (keep it working, minimal changes)

- [ ] **4.2.1** `mvpa2/testing/sweep.py` — only change is that `SkipTest` now comes
  from `unittest` via `mvpa2.testing.tools`. Verify `except SkipTest` still works.
  **No test file changes needed** — `@sweepargs` continues to work as-is.
- [ ] Run: `pytest mvpa2/base/tests/test_collections.py` (uses sweepargs heavily)
- [ ] Commit: `RF: update sweep.py SkipTest import for pytest compatibility`

### 4.3 Fix direct nose imports in test/source files

- [ ] **4.3.1** `mvpa2/tests/test_viz.py` — `from nose.tools import *` → `from mvpa2.testing import *`
- [ ] **4.3.2** `mvpa2/tests/test_openfmri.py` — `from nose.tools import assert_greater` → `from mvpa2.testing import assert_greater`
- [ ] **4.3.3** `mvpa2/tests/test_group_clusterthr.py` — `from nose.tools import assert_greater_equal, assert_greater` → `from mvpa2.testing import ...`
- [ ] **4.3.4** `mvpa2/misc/tests/test_dcov.py` — `from nose.tools import ok_` → `from mvpa2.testing import ok_`
- [ ] **4.3.5** `mvpa2/clfs/sg/__init__.py` — `from nose.plugins.skip import SkipTest` → `from unittest import SkipTest`
- [ ] **4.3.6** `mvpa2/cmdline/cmd_exec.py` — remove `import nose.tools as nt`
- [ ] Commit: `RF: replace direct nose imports in test/source files`

### 4.4 Fix module-level `raise SkipTest` (pytest doesn't support this)

- [ ] **4.4.1** `mvpa2/datasets/tests/test_niftidataset.py` — replace `raise SkipTest` with `pytest.skip("...", allow_module_level=True)`
- [ ] **4.4.2** `mvpa2/tests/test_giftidataset.py` — same
- [ ] Commit: `BF: use pytest.skip(allow_module_level=True) for module-level skips`

### 4.5 Create `conftest.py`

- [ ] **4.5.1** Root `conftest.py`:
  - [ ] Session-scoped autouse fixture wrapping `mvpa2.setup_module()` / `teardown_module()` logic
  - [ ] `collect_ignore` list for `mvpa2/tests/__init__.py`, `mvpa2/tests/runner.py`
  - [ ] Register custom markers: `labile`, `slow`, `ai_generated`
- [ ] Commit: `ENH: add conftest.py with session setup and marker registration`

### 4.6 Create `tox.ini`

- [ ] **4.6.1** `[tox]` envlist: `py3{9,10,11,12,13},lint`
- [ ] **4.6.2** `[testenv]`: `deps = .[test]`, `commands = pytest {posargs:mvpa2/}`
- [ ] **4.6.3** `[testenv:lint]`: `deps = .[devel]`, `commands = ruff check mvpa2/`
- [ ] Run: `tox -e py3` to verify
- [ ] Commit: `ENH: add tox.ini for test automation`

### 4.7 Clean up old test infrastructure

- [ ] **4.7.1** `mvpa2/tests/__init__.py` — remove `run_tests_using_nose()`, simplify `run()`
- [ ] **4.7.2** `mvpa2/tests/test_suite.py` — review; possibly remove if pytest discovery suffices
- [ ] **4.7.3** `mvpa2/base/externals.py` — remove `'nose'` from `_KNOWN` externals
- [ ] **4.7.4** Remove nose from any requirements/dependencies
- [ ] Commit: `RF: remove nose-specific test infrastructure`

### 4.8 Complete test file moves (remaining ~20 in `mvpa2/tests/`)

Each move should be its own commit (or batched by target module) with tests
verified to still pass after the move. Use `git mv` for clean history.

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

## Phase 5: CI Modernization — Travis → GitHub Actions

- [ ] **5.1** Create `.github/workflows/test.yml`:
  - [ ] Matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13
  - [ ] Steps: checkout, setup-python, `pip install -e ".[ci]"`, `tox`
  - [ ] Separate lint job
- [ ] **5.2** Add `[gh-actions]` mapping in `tox.ini` (tox-gh-actions)
- [ ] **5.3** Remove `.travis.yml`
- [ ] Commit: `RF: replace Travis CI with GitHub Actions`

## Phase 6: Dependency and API Modernization

### 6.1 Remove dead dependency support code

- [ ] **6.1.1** Remove Shogun support entirely:
  - [ ] `mvpa2/clfs/sg/` — entire directory (svm.py, sens.py, __init__.py)
  - [ ] Remove all `shogun`/`sg*` entries from `mvpa2/base/externals.py`
  - [ ] Remove shogun entries from `mvpa2/clfs/warehouse.py`
  - [ ] Remove shogun from `mvpa2/testing/clfs.py`
  - [ ] Remove `mvpa2/kernels/sg.py`
  - [ ] Remove related tests: `test_svmkernels.py` shogun parts, `test_kernel.py` shogun parts
  - [ ] Run: `pytest mvpa2/clfs/tests/ mvpa2/kernels/tests/`
  - [ ] Commit: `RF: remove Shogun support (project unmaintained since 2019)`

- [ ] **6.1.2** Remove MDP support:
  - [ ] `mvpa2/mappers/mdp_adaptor.py` — remove
  - [ ] `mvpa2/mappers/lle.py` — remove (MDP-based LLE; sklearn has LLE)
  - [ ] Remove `mdp`/`mdp ge 2.4` entries from externals
  - [ ] Remove related tests
  - [ ] Commit: `RF: remove MDP support (project unmaintained since 2020)`

- [ ] **6.1.3** Remove pprocess support:
  - [ ] Remove pprocess codepaths in `mvpa2/measures/searchlight.py`, `misc/surfing/surf_voxel_selection.py`, `algorithms/searchlight_hyperalignment.py`
  - [ ] Remove `pprocess` from externals
  - [ ] Commit: `RF: remove pprocess support (use joblib instead)`

- [ ] **6.1.4** Remove OpenOpt support:
  - [ ] Remove openopt codepaths in `mvpa2/clfs/gpr.py`, `model_selector.py`
  - [ ] Remove `openopt` from externals
  - [ ] Commit: `RF: remove OpenOpt support (project dead since ~2014)`

- [ ] **6.1.5** Remove scipy.weave/weave support:
  - [ ] Remove weave codepath in `mvpa2/clfs/distance.py`
  - [ ] Remove `weave`/`scipy.weave` from externals
  - [ ] Commit: `RF: remove scipy.weave support (removed from scipy in 1.0)`

- [ ] **6.1.6** Clean up remaining dead externals:
  - [ ] Remove `cPickle`, `hcluster`, `nose`, `mock` (use `unittest.mock`) from externals
  - [ ] Commit: `RF: remove dead external checks`

### 6.2 API modernization

- [ ] **6.2.1** `inspect.getargspec()` → `inspect.getfullargspec()`:
  - [ ] Already shimmed in `mvpa2/support/utils.py`
  - [ ] Ensure all callers use the shim (grep for `getargspec` outside `support/utils.py`)
  - [ ] Commit: `RF(PY3): ensure all getargspec callers use PY3 shim`

- [ ] **6.2.2** `numpy.testing.decorators` → `pytest.mark.skipif`
  - [ ] Commit: `RF: replace deprecated numpy.testing.decorators with pytest.mark`

- [ ] **6.2.3** `collections.MutableMapping` → `collections.abc.MutableMapping` (if any)
- [ ] **6.2.4** `imp` module → `importlib` (if any)

- [ ] **6.2.5** h5py 3.x compatibility:
  - [ ] Remove `import h5py.highlevel` (gone in h5py 3.0)
  - [ ] **⚠ CONSULT**: h5py 3.0 changed string handling — strings stored as bytes
    by default. This affects how PyMVPA's custom HDF5 serialization reads/writes
    string attributes. The `ensure_str()` helper (from branch WIP) addresses this
    but needs thorough testing with real saved datasets from PY2 era.
  - [ ] Commit: `RF: update for h5py 3.x API (remove highlevel, fix string handling)`

- [ ] **6.2.6** scikit-learn API updates:
  - [ ] Check for deprecated parameter names, removed methods
  - [ ] Commit: `RF: update for scikit-learn 1.x API changes`

- [ ] **6.2.7** `griddata`:
  - [ ] Replace `matplotlib.mlab.griddata` (removed mpl 3.1) with `scipy.interpolate.griddata`
  - [ ] Update `mvpa2/support/griddata.py`
  - [ ] Commit: `RF: use scipy.interpolate.griddata (mpl.mlab.griddata removed)`

- [ ] **6.2.8** Version constraints in pyproject.toml:
  - [ ] `numpy>=1.22`, `scipy>=1.7`, `h5py>=3.0`, `nibabel>=3.0`, `scikit-learn>=1.0`

## Phase 7: Cleanup and Polish

- [ ] **7.1** Final grep: no remaining `from future`, `from past`, `from builtins`, `from six`
- [ ] **7.2** Final grep: no remaining `from nose`, `import nose`
- [ ] **7.3** Update `mvpa2/base/info.py` version string
- [ ] **7.4** Run `ruff check --fix` on entire codebase
  - [ ] Use `datalad run` to record: `datalad run -m "STY: ruff auto-fix" -- ruff check --fix mvpa2/`
- [ ] **7.5** Run `codespell` (with `ignore-regex = https?://\S+` to protect URLs)
  - [ ] Use `datalad run` to record
- [ ] **7.6** Verify full test suite: `tox -e py3`
- [ ] **7.7** Remove stale files:
  - [ ] `Changelog.orig`, `Changelog.rej`
  - [ ] Old `.so` files
  - [ ] `py3tool.py` (confirm gone)
  - [ ] `mvpa2/tests/runner.py` (if no longer needed)
- [ ] **7.8** Commit: `STY: final cleanup and polish`

---

## Execution Order & Dependencies

```
Phase 0 (cleanup WIP — stash + gitignore)
  ↓
Phase 1 (build system)  ←→  Phase 2 (strip future/six)
  ↓                            ↓
         Phase 3 (convert remaining files)
              ↓
         Phase 4 (pytest migration)
              ↓
         Phase 5 (CI)
              ↓
         Phase 6 (dep modernization)
              ↓
         Phase 7 (polish)
```

Phases 1 and 2 can overlap. Phase 3 can start after Phase 2 patterns are
established. Phase 4 can start once Phase 3 syntax fixes are done (tests
need to at least parse under PY3). Phases 5-7 are sequential cleanup.

## Consultation Checkpoints (⚠ CONSULT)

These steps involve non-obvious semantic decisions where the implementer
should stop and consult the maintainer:

1. **Phase 2.6** — `old_div()` replacement: integer vs float division ambiguity
2. **Phase 2.7** — `with_metaclass` removal: HDF5 serialization may depend on
   `__metaclass__` class attribute presence
3. **Phase 2.9** — PY2 guards in `hdf5.py`: must preserve ability to load
   PY2-era saved HDF5 files while removing PY2 saving codepaths
4. **Phase 6.2.5** — h5py 3.x string handling: custom serialization may need
   careful testing with real PY2-era datasets
5. **Any step** where tests fail in non-obvious ways after a mechanical
   transformation — bytes/str confusion, changed iteration order, or
   numeric precision differences between PY2 and PY3

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| C extensions may not build with meson-python | Use pre-generated SWIG wrapper (already committed); test early in Phase 1 |
| `sweepargs` behavior differences under pytest | Keep `sweepargs` as-is; only fix `SkipTest` import; defer full parametrize conversion |
| `old_div()` replacement ambiguity (int vs float division) | ⚠ CONSULT — check each usage individually |
| Deep PY2 bytes/str assumptions in NIML/HDF5 code | ⚠ CONSULT — handle case-by-case; test with real PY2-era data files |
| `support/_copy.py` (vendored PY2 copy.py) may serve specific purpose | Investigate callers before replacing with stdlib `copy` |
| `assert_raises` behavior change (nose's vs pytest.raises) | pytest.raises works as context manager AND callable — datalad proved this works |
| Removing Shogun/MDP/pprocess may break user workflows | Announce deprecation in changelog; these projects are dead anyway |
| HDF5 files saved under PY2 may not load under PY3 | Keep `builtins_mods` tuple for backward-compat loading; test with saved fixtures |
