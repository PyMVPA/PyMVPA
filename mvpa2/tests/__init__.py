# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit test interface for PyMVPA"""

from mvpa2.base import externals


def collect_unit_tests(verbosity=1):
    """Return list of legacy unittest-style modules under ``mvpa2.tests``.

    Kept for backward-compatibility with external scripts that introspect
    the test layout. Pytest discovers tests directly via ``pytest mvpa2/``.
    """
    tests = [
        'test_externals',
        'test_dochelpers',
        'test_som',
        'test_state',
        'test_params',
        'test_config',
        'test_support',
        'test_verbosity',
        'test_report',
        'test_cmdline',
        'test_args',
        'test_meg',
        'test_clf',
        'test_regr',
        'test_knn',
        'test_gnb',
        'test_svm',
        'test_plr',
        'test_smlr',
        'test_svdmapper',
        'test_procrust',
        'test_hyperalignment',
        'test_benchmarks_hyperalignment',
        'test_searchlight_hyperalignment',
        'test_transformers',
        'test_searchlight',
        'test_rfe',
        'test_ifs',
        'test_perturbsensana',
        'test_winner',
        'test_suite',
    ]

    optional = [('scipy', 'ridge'), ('scipy', 'gpr')]
    for external, testname in optional:
        if externals.exists(external):
            tests.append('test_%s' % testname)
    return tests


def run(limit=None, verbosity=None, exit_=False):
    """Run the PyMVPA test suite via pytest.

    Parameters
    ----------
    limit : None or list
      If None, run the whole ``mvpa2`` test tree. Otherwise a list of
      pytest ``-k`` expressions or test paths.
    verbosity : None or int
      Forwarded as ``-v`` (>=1) / ``-vv`` (>=2).
    exit_ : bool, optional
      Raise ``SystemExit`` on the test exit code.
    """
    import pytest

    args = ['mvpa2']
    if verbosity is not None and verbosity >= 2:
        args.insert(0, '-vv')
    elif verbosity is not None and verbosity >= 1:
        args.insert(0, '-v')
    if limit:
        for entry in limit:
            args.extend(['-k', entry])

    rc = pytest.main(args)
    if exit_:
        raise SystemExit(rc)
    return rc


# avoid pytest collecting these helpers as tests
collect_unit_tests.__test__ = False
run.__test__ = False
__test__ = False

if __name__ == "__main__":  # pragma: no cover
    run(exit_=True, verbosity=1)
