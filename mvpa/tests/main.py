# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit test console interface for PyMVPA"""

import unittest
import sys
import numpy as np

from mvpa import _random_seed, cfg
from mvpa.base import externals, warning
from mvpa.tests import collectTestSuites


def main():
    if __debug__:
        from mvpa.base import debug
        # Lets add some targets which provide additional testing
        debug.active += ['CHECK_.*']
        # NOTE: it had to be done here instead of test_clf.py for
        # instance, since for CHECK_RETRAIN it has to be set before object
        # gets created, ie while importing clfs.warehouse

    suites = collectTestSuites()

    # and make global test suite
    ts = unittest.TestSuite(suites.values())


    class TextTestRunnerPyMVPA(unittest.TextTestRunner):
        """Extend TextTestRunner to print out random seed which was
        used in the case of failure"""
        def run(self, test):
            result = super(TextTestRunnerPyMVPA, self).run(test)
            if not result.wasSuccessful():
                print "MVPA_SEED=%s" % _random_seed
                sys.exit(1)
            return result

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
        np.seterr(**np_errsettings)


if __name__ == '__main__':
    main()
