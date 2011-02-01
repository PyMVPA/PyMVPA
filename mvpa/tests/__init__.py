# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit test interface for PyMVPA"""

import unittest
import numpy as np
from mvpa import _random_seed, cfg
from mvpa.base import externals, warning


def collectTestSuites():
    """Runs over all tests it knows and composes a dictionary with test suite
    instances as values and IDs as keys. IDs are the filenames of the unittest
    without '.py' extension and 'test_' prefix.

    During collection this function will run a full and verbose test for all
    known externals.
    """
    # list all test modules (without .py extension)
    tests = [
        # Basic data structures/manipulators
        'test_externals',
        'test_base',
        'test_dochelpers',
        'test_dataset',
        'test_arraymapper',
        'test_boxcarmapper',
        'test_som',
        'test_neighbor',
        'test_maskeddataset',
        'test_metadataset',
        'test_splitter',
        'test_state',
        'test_params',
        'test_eepdataset',
        # Misc supporting utilities
        'test_config',
        'test_stats',
        'test_support',
        'test_verbosity',
        'test_iohelpers',
        'test_report',
        'test_datasetfx',
        'test_cmdline',
        'test_args',
        'test_eepdataset',
        'test_meg',
        # Classifiers (longer tests)
        'test_kernel',
        'test_clf',
        'test_regr',
        'test_knn',
        'test_gnb',
        'test_svm',
        'test_plr',
        'test_smlr',
        # Various algorithms
        'test_svdmapper',
        'test_procrust',
        'test_hyperalignment',
        'test_samplegroupmapper',
        'test_transformers',
        'test_transerror',
        'test_clfcrossval',
        'test_searchlight',
        'test_rfe',
        'test_ifs',
        'test_datameasure',
        'test_perturbsensana',
        'test_splitsensana',
        # And the suite (all-in-1)
        'test_suite',
        ]

    # provide people with a hint about the warnings that might show up in a
    # second
    warning('Testing for availability of external software packages. Test '
            'cases depending on missing packages will not be part of the test '
            'suite.')

    # So we could see all warnings about missing dependencies
    warning.maxcount = 1000
    # fully test of externals
    externals.testAllDependencies()


    __optional_tests = [ ('scipy', 'ridge'),
                         ('scipy', 'stats_sp'),
                         ('scipy', 'datasetfx_sp'),
                         (['lars','scipy'], 'lars'),
                         ('nifti', 'niftidataset'),
                         ('mdp', 'icamapper'),
                         ('scipy', 'zscoremapper'),
                         ('pywt', 'waveletmapper'),
                         (['cPickle', 'gzip'], 'hamster'),
    #                     ('mdp', 'pcamapper'),
                       ]

    if not cfg.getboolean('tests', 'lowmem', default='no'):
        __optional_tests += [(['nifti', 'lxml'], 'atlases')]


    # and now for the optional tests
    optional_tests = []

    for external, testname in __optional_tests:
        if externals.exists(external):
            optional_tests.append('test_%s' % testname)


    # finally merge all of them
    tests += optional_tests

    # import all test modules
    for t in tests:
        exec 'import ' + t

    # instanciate all tests suites and return dict of them (with ID as key)
    return dict([(t[5:], eval(t + '.suite()')) for t in tests ])



def run(limit=None, verbosity=None):
    """Runs the full or a subset of the PyMVPA unittest suite.

    :Parameters:
      limit: None | list
        If None, the full test suite is run. Alternatively, a list with test IDs
        can be provides. IDs are the base filenames of the test implementation,
        e.g. the ID for the suite in 'mvpa/tests/test_niftidataset.py' is
        'niftidataset'.
      verbosity: None | int
        Verbosity of unittests execution. If None, controlled by PyMVPA
        configuration tests/verbosity.  Values higher than 2 enable all Python, 
        NumPy and PyMVPA warnings
    """
    if __debug__:
        from mvpa.base import debug
        # Lets add some targets which provide additional testing
        debug.active += ['CHECK_.*']

    # collect all tests
    suites = collectTestSuites()

    if limit is None:
        # make global test suite (use them all)
        ts = unittest.TestSuite(suites.values())
    else:
        ts = unittest.TestSuite([suites[s] for s in limit])


    class TextTestRunnerPyMVPA(unittest.TextTestRunner):
        """Extend TextTestRunner to print out random seed which was
        used in the case of failure"""
        def run(self, test):
            """Run the bloody test and puke the seed value if failed"""
            result = super(TextTestRunnerPyMVPA, self).run(test)
            if not result.wasSuccessful():
                print "MVPA_SEED=%s" % _random_seed

    if verbosity is None:
        verbosity = int(cfg.get('tests', 'verbosity', default=1))

    if verbosity < 3:
        # no MVPA warnings during whole testsuite (but restore handlers later on)
        handler_backup = warning.handlers
        warning.handlers = []

        # No python warnings (like ctypes version for slmr)
        import warnings
        warnings.simplefilter('ignore')

        # No numpy
        np_errsettings = np.geterr()
        np.seterr(**dict([(x, 'ignore') for x in np_errsettings]))

    # finally run it
    TextTestRunnerPyMVPA(verbosity=verbosity).run(ts)

    if verbosity < 3:
        # restore warning handlers
        warning.handlers = handler_backup
        np.seterr(np_errsettings)
