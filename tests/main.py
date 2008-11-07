#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Main unit test interface for PyMVPA"""

import unittest
import sys

from mvpa import _random_seed
from mvpa.base import externals, warning

if __debug__:
    from mvpa.base import debug
    # Lets add some targets which provide additional testing
    debug.active += ['CHECK_.*']
    # NOTE: it had to be done here instead of test_clf.py for
    # instance, since for CHECK_RETRAIN it has to be set before object
    # gets created, ie while importing clfs.warehouse

# list all test modules (without .py extension)
tests = [
    # Basic data structures/manipulators
    'test_dataset',
    'test_arraymapper',
    'test_boxcarmapper',
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
    'test_svm',
    'test_plr',
    'test_smlr',
    # Various algorithms
    'test_svdmapper',
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

# So we could see all warnings about missing dependencies
warning.maxcount = 1000
# fully test of externals
externals.testAllDependencies()


__optional_tests = ( ('scipy', 'ridge'),
                     ('scipy', 'datasetfx_sp'),
                     (['lars','scipy'], 'lars'),
                     ('nifti', 'niftidataset'),
                     ('mdp', 'icamapper'),
                     ('pywt', 'waveletmapper'),
                     (['cPickle', 'gzip'], 'hamster'),
#                     ('mdp', 'pcamapper'),
                     )

# and now for the optional tests
optional_tests = []

for external, testname in __optional_tests:
    if externals.exists(external):
        optional_tests.append('test_%s' % testname)


# finally merge all of them
tests += optional_tests

# No python warnings (like ctypes version for slmr)
import warnings
warnings.simplefilter('ignore')

# import all test modules
for t in tests:
    exec 'import ' + t

# no MVPA warnings during whole testsuite
from mvpa.base import warning
warning.handlers = []

def main():
    # load all tests suites
    suites = [ eval(t + '.suite()') for t in tests ]

    # and make global test suite
    ts = unittest.TestSuite(suites)


    class TextTestRunnerPyMVPA(unittest.TextTestRunner):
        """Extend TextTestRunner to print out random seed which was
        used in the case of failure"""
        def run(self, test):
            result = super(TextTestRunnerPyMVPA, self).run(test)
            if not result.wasSuccessful():
                print "MVPA_SEED=%s" % _random_seed
                sys.exit(1)
            return result

    # finally run it
    TextTestRunnerPyMVPA().run(ts)

if __name__ == '__main__':
    main()

