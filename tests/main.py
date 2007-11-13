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
    'test_nfoldsplitter',
    # Misc supporting utilities
    'test_stats',
    'test_support',
    'test_verbosity',
    'test_iohelpers',
    # Classifiers (longer tests)
    'test_knn',
    'test_svm',
    'test_clfcrossval',
    'test_searchlight',
    ]
#          'test_plf',
#          'test_ifs',
#          'test_rfe',


# import all test modules
for t in tests:
    exec 'import ' + t


if __name__ == '__main__':

    # load all tests suites
    suites = [ eval(t + '.suite()') for t in tests ]

    # and make global test suite
    ts = unittest.TestSuite( suites )

    # finally run it
    unittest.TextTestRunner().run( ts )

