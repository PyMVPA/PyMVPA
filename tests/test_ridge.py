#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ridge regression classifier"""

from mvpa.clfs.ridge import RidgeReg
from scipy.stats import pearsonr
from tests_warehouse import *

class RidgeRegTests(unittest.TestCase):

    def testRidgeReg(self):
        # not the perfect dataset with which to test, but
        # it will do for now.
        data = datasets['dumb']

        clf = RidgeReg()

        clf.train(data)

        # prediction has to be almost perfect
        # test with a correlation
        pre = clf.predict(data.samples)
        cor = pearsonr(pre,data.labels)
        self.failUnless(cor[0] > .8)

    def testRidgeRegState(self):
        data = datasets['dumb']

        clf = RidgeReg()

        clf.train(data)

        clf.states.enable('predictions')

        p = clf.predict(data.samples)

        self.failUnless((p == clf.predictions).all())


def suite():
    return unittest.makeSuite(RidgeRegTests)


if __name__ == '__main__':
    import runner

