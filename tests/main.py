#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Main unit test interface for PyMVPA"""

import unittest

# list all test modules (without .py extension)
tests = [ 'test_dataset',
          'test_maskmapper',
          'test_neighbor',
          'test_maskeddataset',
          'test_nfoldsplitter',
          'test_knn',
          'test_clfcrossval'
          ]
#          'test_algorithms',
#          'test_svm',
#          'test_plf',
#          'test_xvalpattern',
#          'test_searchlight',
#          'test_ifs',
#          'test_rfe',
#          'test_featsel',
#          'test_support',
#          'test_stats' ]


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

