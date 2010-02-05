# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA logistic regression classifier"""

from mvpa.clfs.plr import PLR
from tests_warehouse import *


class PLRTests(unittest.TestCase):

    def test_plr(self):
        data = datasets['dumb2']

        clf = PLR()

        clf.train(data)

        # prediction has to be perfect
        self.failUnless((clf.predict(data.samples) == data.targets).all())

    def test_plr_state(self):
        data = datasets['dumb2']

        clf = PLR()

        clf.train(data)

        clf.states.enable('estimates')
        clf.states.enable('predictions')

        p = clf.predict(data.samples)

        self.failUnless((p == clf.states.predictions).all())
        self.failUnless(N.array(clf.states.estimates).shape == N.array(p).shape)


def suite():
    return unittest.makeSuite(PLRTests)


if __name__ == '__main__':
    import runner

