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
- **Minimize diff**: adopt datalad-style nose compatibility shims so test files
  need minimal changes (keep `ok_`, `eq_`, `assert_equal`, `sweepargs`, etc.
  as thin wrappers — do NOT rewrite individual test assertions to bare `assert`)

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

- [ ] **0.1** Triage uncommitted working tree changes (8 files):
  - [ ] Stage & commit useful changes:
    - [ ] `mvpa2/base/collections.py` — bytes→str decode for `doc` parameter
    - [ ] `mvpa2/base/hdf5.py` — `builtins_mods` tuple, `ensure_str()` usage, optional `h5py.highlevel`
    - [ ] `mvpa2/datasets/sources/skl_data.py` — use `getargspec` shim
    - [ ] `mvpa2/support/utils.py` — `getargspec()`, `ensure_str()` helpers
    - [ ] `mvpa2/misc/support.py` — remove old `builtins_mod`
    - [ ] `mvpa2/clfs/transerror.py` — cosmetic label fix
  - [ ] Discard debug artifacts:
    - [ ] `mvpa2/tests/test_transerror.py` — remove `pdb.set_trace()`, `i = 1`, `pl.show()`
    - [ ] `mvpa2/base/tests/test_hdf5.py` — remove hardcoded `testfile = test_files[0]`, dead code after `raise`
- [ ] **0.2** Confirm `py3tool.py` is deleted (already removed on branch)
- [ ] **0.3** Remove pre-built `.so` files for old Python versions from `mvpa2/clfs/lib*/`
- [ ] **0.4** Add/update `.gitignore`:
  - [ ] `*.so`, `*.pyc`, `__pycache__/`
  - [ ] `*.egg-info/`, `.eggs/`
  - [ ] `.cache/`, `.hypothesis/`, `.pytest_cache/`
  - [ ] `.tox/`, `.venv/`
  - [ ] `.npm/`

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

## Phase 2: Strip `future`/`six`/`past` Library — Pure Python 3

Bulk removal across 89 already-converted files, plus replacement patterns.

- [ ] **2.1** Remove `from __future__ import` lines (all files):
  - [ ] `division`
  - [ ] `print_function`
  - [ ] `unicode_literals`
  - [ ] `absolute_import`
- [ ] **2.2** Remove `from builtins import` lines:
  - [ ] `str`, `range`, `zip`, `map`, `filter`, `next`, `object`, `bytes`
- [ ] **2.3** Remove `from past.builtins import basestring`:
  - [ ] Replace all `basestring` usage with `str`
  - [ ] Files: `mvpa2/base/collections.py`, `mvpa2/base/dataset.py`, `mvpa2/base/dochelpers.py`, `mvpa2/base/state.py`, `mvpa2/clfs/meta.py`, `mvpa2/mappers/base.py`, `mvpa2/misc/support.py`, and ~20 others
- [ ] **2.4** Remove `from past.builtins import cmp`:
  - [ ] Provide `cmp` as a utility: `cmp = lambda a, b: (a > b) - (a < b)` in `mvpa2/misc/support.py`
  - [ ] Update callers to import from there, or inline
- [ ] **2.5** Remove `from past.utils import old_div`:
  - [ ] Replace `old_div(a, b)` with `a // b` or `a / b` (check each — integer vs float division)
  - [ ] Affected files: `mvpa2/atlases/base.py`, `mvpa2/atlases/transformation.py`, `mvpa2/misc/support.py`, `mvpa2/misc/io/base.py`, and others
- [ ] **2.6** Remove `from future import standard_library; standard_library.install_aliases()`:
  - [ ] All files using this — stdlib names are already correct in PY3
- [ ] **2.7** Remove `from future.utils import with_metaclass`:
  - [ ] Replace `class Foo(with_metaclass(Meta, Base)):` → `class Foo(Base, metaclass=Meta):`
- [ ] **2.8** Remove `from six import reraise`:
  - [ ] Replace with native `raise exc.with_traceback(tb)` in `mvpa2/testing/sweep.py`
- [ ] **2.9** Remove PY2 guards:
  - [ ] `PY2 = sys.version_info[0] == 2` in `mvpa2/misc/support.py` and all `if PY2:` branches
  - [ ] PY2-specific codepaths in `mvpa2/base/hdf5.py`
- [ ] **2.10** Remove `mvpa2/support/_copy.py`:
  - [ ] Find all imports: `from mvpa2.support._copy import deepcopy` → `from copy import deepcopy`
- [ ] **2.11** Remove `future` and `six` from dependencies/externals:
  - [ ] `mvpa2/base/externals.py` — remove `'future'`, `'six'` entries
  - [ ] Remove `cPickle` from `_KNOWN` externals (it's `pickle` in PY3)
- [ ] **2.12** Clean up `SyntaxError` suppression:
  - [ ] `mvpa2/base/externals.py` — remove `'SyntaxError'` from caught exceptions (was added as TODO during 2to3)

## Phase 3: Convert Remaining ~53 Unconverted Source Files

All mechanical PY3 fixes. Patterns:
`xrange→range`, `.iteritems()→.items()`, `.itervalues()→.values()`,
`.iterkeys()→.keys()`, `print x→print(x)`, `raise X, msg→raise X(msg)`,
`except X, e→except X as e`, `basestring→str`, `execfile(f)→exec(open(f).read())`,
`has_key(k)→k in d`

### 3a. `support/nibabel/` (8 files, 43 PY2 hits)
- [ ] `afni_niml.py` — 14 hits (xrange, iteritems, basestring)
- [ ] `afni_niml_dset.py` — 10 hits (xrange, iteritems, basestring)
- [ ] `afni_suma_spec.py` — 7 hits (print, xrange, iteritems, basestring)
- [ ] `afni_niml_annot.py` — 5 hits (xrange, basestring)
- [ ] `afni_suma_1d.py` — 2 hits (xrange, basestring)
- [ ] `afni_niml_roi.py` — 1 hit (basestring)
- [ ] `surf_fs_asc.py` — 2 hits (xrange)
- [ ] `surf_gifti.py` — 2 hits (iteritems)

### 3b. `support/afni/` (5 files, 58 PY2 hits)
- [ ] `lib_prep_afni_surf.py` — 29 hits (27 print-stmts, 2 iteritems)
- [ ] `afni_surface_alphasim.py` — 12 hits (7 print, 5 xrange)
- [ ] `lib_plot_slices.py` — 7 hits (print, xrange, basestring)
- [ ] `afni_utils.py` — 6 hits (5 print, 1 iteritems)
- [ ] `suma_setup_instacorr.py` — 4 hits (print)

### 3c. `misc/surfing/` (4 files, 27 PY2 hits)
- [ ] `volume_mask_dict.py` — 8 hits (iteritems, basestring)
- [ ] `volsurf.py` — 8 hits (xrange, iteritems)
- [ ] `surf_voxel_selection.py` — 8 hits (xrange, iteritems, basestring)
- [ ] `queryengine.py` — 3 hits (xrange, iteritems)

### 3d. `cmdline/` (7 files, 14 PY2 hits)
- [ ] `cmd_crossval.py` — 5 hits (print, xrange)
- [ ] `cmd_describe.py` — 2 hits (print)
- [ ] `cmd_dump.py` — 1 hit (print)
- [ ] `cmd_exec.py` — 1 hit (execfile) + has `import nose.tools as nt`
- [ ] `cmd_hyperalign.py` — 1 hit (print)
- [ ] `cmd_info.py` — 2 hits (print)
- [ ] `cmd_mkevds.py` — 2 hits (xrange, iteritems)

### 3e. `datasets/` (5 files, 27 PY2 hits)
- [ ] `cosmo.py` — 12 hits (iteritems, basestring)
- [ ] `niml.py` — 6 hits (iteritems, basestring)
- [ ] `formats.py` — 4 hits (raise-comma — **syntax error in PY3**)
- [ ] `gifti.py` — 3 hits (basestring)
- [ ] `eeglab.py` — 2 hits (xrange, basestring)

### 3f. `measures/` (3 files, 7 PY2 hits)
- [ ] `corrcoef.py` — 1 hit (xrange)
- [ ] `noiseperturbation.py` — 1 hit (xrange)
- [ ] `winner.py` — 5 hits (print, xrange, iteritems)

### 3g. `algorithms/` (2 files, 5 PY2 hits)
- [ ] `group_clusterthr.py` — 4 hits (xrange)
- [ ] `searchlight_hyperalignment.py` — 1 hit (print)

### 3h. Other scattered files (8 files, ~30 PY2 hits)
- [ ] `support/_copy.py` — 20 hits (consider replacing with stdlib `copy`)
- [ ] `support/bayes/multivariate_polya.py` — 8 hits (print in `__main__`)
- [ ] `support/bayes/partial_independence.py` — 6 hits (print in `__main__`)
- [ ] `misc/plot/flat_surf.py` — 2 hits (basestring)
- [ ] `mappers/som.py` — 2 hits (xrange, raise-comma — **syntax error in PY3**)
- [ ] `clfs/model_selector.py` — 2 hits (print)
- [ ] `viz/__init__.py` — 2 hits (basestring)
- [ ] `support/ipython/ipy_pymvpa_completer.py` — 1 hit (print)

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

### 3j. Fix `test_suite.py` PY2 syntax
- [ ] `exec "from mvpa2.suite import *"` → `exec("from mvpa2.suite import *")`
- [ ] `.iteritems()` → `.items()`

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
  assert_true = ok_
  assert_false = lambda val, msg=None: (assert not val, msg)  # or proper def
  assert_equal = eq_
  assert_equals = eq_
  assert_not_equal = lambda a, b, msg=None: assert a != b, msg
  assert_not_equals = assert_not_equal
  assert_in = lambda a, b, msg=None: assert a in b, msg
  assert_not_in = lambda a, b, msg=None: assert a not in b, msg
  assert_is_instance = lambda a, b, msg=None: assert isinstance(a, b), msg
  assert_greater = lambda a, b, msg=None: assert a > b, msg
  assert_greater_equal = lambda a, b, msg=None: assert a >= b, msg
  ```
- [ ] **4.1.4** `assert_raises = pytest.raises` (direct alias, like datalad)
- [ ] **4.1.5** Replace all 4 `make_decorator` uses with `functools.wraps`:
  - [ ] `with_tempfile` (line ~329)
  - [ ] `reseed_rng` (line ~361)
  - [ ] `nodebug` (line ~404)
  - [ ] `labile` (line ~463)
- [ ] **4.1.6** Replace `skipif` from `numpy.testing.dec` with `pytest.mark.skipif`
- [ ] **4.1.7** Drop unused imports: `timed`, `with_setup`, `raises`, `istest`, `nottest`

### 4.2 Fix `sweepargs` (keep it working, minimal changes)

- [ ] **4.2.1** `mvpa2/testing/sweep.py` — only change is that `SkipTest` now comes
  from `unittest` via `mvpa2.testing.tools`. Verify `except SkipTest` still works.
  **No test file changes needed** — `@sweepargs` continues to work as-is.

### 4.3 Fix direct nose imports in test/source files

- [ ] **4.3.1** `mvpa2/tests/test_viz.py` — `from nose.tools import *` → `from mvpa2.testing import *`
- [ ] **4.3.2** `mvpa2/tests/test_openfmri.py` — `from nose.tools import assert_greater` → `from mvpa2.testing import assert_greater`
- [ ] **4.3.3** `mvpa2/tests/test_group_clusterthr.py` — `from nose.tools import assert_greater_equal, assert_greater` → `from mvpa2.testing import ...`
- [ ] **4.3.4** `mvpa2/misc/tests/test_dcov.py` — `from nose.tools import ok_` → `from mvpa2.testing import ok_`
- [ ] **4.3.5** `mvpa2/clfs/sg/__init__.py` — `from nose.plugins.skip import SkipTest` → `from unittest import SkipTest`
- [ ] **4.3.6** `mvpa2/cmdline/cmd_exec.py` — remove `import nose.tools as nt`

### 4.4 Fix module-level `raise SkipTest` (pytest doesn't support this)

- [ ] **4.4.1** `mvpa2/datasets/tests/test_niftidataset.py` — replace `raise SkipTest` with `pytest.skip("...", allow_module_level=True)`
- [ ] **4.4.2** `mvpa2/tests/test_giftidataset.py` — same

### 4.5 Create `conftest.py`

- [ ] **4.5.1** Root `conftest.py`:
  - [ ] Session-scoped autouse fixture wrapping `mvpa2.setup_module()` / `teardown_module()` logic
  - [ ] `collect_ignore` list for `mvpa2/tests/__init__.py`, `mvpa2/tests/runner.py`
  - [ ] Register custom markers: `labile`, `slow`, `ai_generated`

### 4.6 Create `tox.ini`

- [ ] **4.6.1** `[tox]` envlist: `py3{9,10,11,12,13},lint`
- [ ] **4.6.2** `[testenv]`: `deps = .[test]`, `commands = pytest {posargs:mvpa2/}`
- [ ] **4.6.3** `[testenv:lint]`: `deps = .[devel]`, `commands = ruff check mvpa2/`

### 4.7 Clean up old test infrastructure

- [ ] **4.7.1** `mvpa2/tests/__init__.py` — remove `run_tests_using_nose()`, simplify `run()`
- [ ] **4.7.2** `mvpa2/tests/test_suite.py` — review; possibly remove if pytest discovery suffices
- [ ] **4.7.3** `mvpa2/base/externals.py` — remove `'nose'` from `_KNOWN` externals
- [ ] **4.7.4** Remove nose from any requirements/dependencies

### 4.8 Complete test file moves (remaining ~20 in `mvpa2/tests/`)

These tests should move to their corresponding module `tests/` subdirectory:

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

## Phase 5: CI Modernization — Travis → GitHub Actions

- [ ] **5.1** Create `.github/workflows/test.yml`:
  - [ ] Matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13
  - [ ] Steps: checkout, setup-python, `pip install -e ".[ci]"`, `tox`
  - [ ] Separate lint job
- [ ] **5.2** Add `[gh-actions]` mapping in `tox.ini` (tox-gh-actions)
- [ ] **5.3** Remove `.travis.yml`

## Phase 6: Dependency and API Modernization

- [ ] **6.1** `inspect.getargspec()` → `inspect.getfullargspec()`:
  - [ ] Already shimmed in `mvpa2/support/utils.py`
  - [ ] Ensure all callers use the shim (grep for `getargspec` outside `support/utils.py`)
- [ ] **6.2** `numpy.testing.decorators` → `pytest.mark.skipif`
- [ ] **6.3** `collections.MutableMapping` → `collections.abc.MutableMapping` (if any)
- [ ] **6.4** `imp` module → `importlib` (if any)
- [ ] **6.5** h5py 3.x compatibility (string handling, no `highlevel`)
- [ ] **6.6** scikit-learn API updates (parameter renames, deprecated methods)
- [ ] **6.7** Version constraints in pyproject.toml:
  - [ ] `numpy>=1.22`, `scipy>=1.7`, `h5py>=3.0`, `nibabel>=3.0`, `scikit-learn>=1.0`
- [ ] **6.8** Remove dead external checks from `mvpa2/base/externals.py`:
  - [ ] `mdp`
  - [ ] `shogun` / `modshogun`
  - [ ] `pprocess`
  - [ ] `openopt`
  - [ ] `weave` / `scipy.weave`
  - [ ] `cPickle`
  - [ ] `nose`

## Phase 7: Cleanup and Polish

- [ ] **7.1** Final grep: no remaining `from future`, `from past`, `from builtins`, `from six`
- [ ] **7.2** Final grep: no remaining `from nose`, `import nose`
- [ ] **7.3** Update `mvpa2/base/info.py` version string
- [ ] **7.4** Run `ruff check --fix` on entire codebase
- [ ] **7.5** Run `codespell` (with `ignore-regex = https?://\S+` to protect URLs)
- [ ] **7.6** Verify full test suite: `tox -e py3`
- [ ] **7.7** Remove stale files:
  - [ ] `Changelog.orig`, `Changelog.rej`
  - [ ] Old `.so` files
  - [ ] `py3tool.py` (confirm gone)
  - [ ] `mvpa2/tests/runner.py` (if no longer needed)

---

## Execution Order & Dependencies

```
Phase 0 (cleanup WIP)
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

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| C extensions may not build with meson-python | Use pre-generated SWIG wrapper (already committed); test early in Phase 1 |
| `sweepargs` behavior differences under pytest | Keep `sweepargs` as-is; only fix `SkipTest` import; defer full parametrize conversion |
| `old_div()` replacement ambiguity (int vs float division) | Check each usage individually; `old_div` was PY2's `/` operator which does int division for ints |
| Deep PY2 bytes/str assumptions in NIML/HDF5 code | Handle case-by-case in Phase 3; HDF5 already partially addressed on branch |
| `support/_copy.py` (vendored PY2 copy.py) may serve specific purpose | Investigate callers before replacing with stdlib `copy` |
| `assert_raises` behavior change (nose's vs pytest.raises) | pytest.raises works as context manager AND callable — datalad proved this works |
