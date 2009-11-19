# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ..."""

# See other tests and test_procrust.py for some example on what to do ;)

from mvpa.algorithms.hyperalignment import Hyperalignment

# Somewhat slow but provides all needed ;)
from tests_warehouse import *

# if you need some classifiers
#from tests_warehouse_clfs import *

class HyperAlignmentTests(unittest.TestCase):


    def testBasicFunctioning(self):
        # TODO
        pass

    def testPossibleInputs(self):
        # get a dataset with a very high SNR
        pass



def suite():
    return unittest.makeSuite(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

