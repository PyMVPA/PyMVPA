### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Main unit test interface for PyMVPA
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import unittest

# list all test modules (without .py extension)
tests = [ 'dataset',
          'maskmapper',
          'maskeddataset',
          'nfoldsplitter',
          'crossval' ]
#          'test_algorithms',
#          'test_knn',
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


