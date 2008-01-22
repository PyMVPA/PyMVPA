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

# list all test modules (without .py extension)
tests = [
    # Basic data structures/manipulators
    'test_dataset',
    'test_maskmapper',
    'test_neighbor',
    'test_maskeddataset',
    'test_niftidataset',
    'test_splitter',
    'test_state',
    # Misc supporting utilities
    'test_stats',
    'test_support',
    'test_verbosity',
    'test_iohelpers',
    'test_signal',
    # Classifiers (longer tests)
    'test_clf',
    'test_knn',
    'test_svm',
    'test_plf',
    'test_ridge',
    # Various algorithms
    'test_transformers',
    'test_transerror',
    'test_clfcrossval',
    'test_searchlight',
    'test_rfe',
    'test_ifs',
    'test_datameasure',
    'test_perturbsensana',
    'test_splitsensana',
    'test_anova',
    'test_nullhyptest'
    ]



# import all test modules
for t in tests:
    exec 'import ' + t

# no warnings during whole testsuite
from mvpa.misc import warning
warning.handlers = []

def main():
    # load all tests suites
    suites = [ eval(t + '.suite()') for t in tests ]

    # and make global test suite
    ts = unittest.TestSuite( suites )

    # finally run it
    unittest.TextTestRunner().run( ts )

if __name__ == '__main__':
    main()

