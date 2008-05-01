#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA sparse multinomial logistic regression classifier"""

import unittest
from mvpa.datasets.dataset import Dataset
from mvpa.clfs.smlr import SMLR
import numpy as N
from mvpa.misc.data_generators import dumbFeatureDataset


class SMLRTests(unittest.TestCase):

    def testSMLR(self):
        data = dumbFeatureDataset()

        clf = SMLR()

        clf.train(data)

        # prediction has to be perfect
        #
        # XXX yoh: whos said that?? ;-)
        #
        # There is always a tradeoff between learning and
        # generalization errors so...  but in this case the problem is
        # more interesting: absent bias disallows to learn data you
        # have here -- there is no solution which would pass through
        # (0,0)
        predictions = clf.predict(data.samples)
        self.failUnless((predictions == data.labels).all())


    def testSMLRState(self):
        data = dumbFeatureDataset()

        clf = SMLR()

        clf.train(data)

        clf.states.enable('values')
        clf.states.enable('predictions')

        p = N.asarray(clf.predict(data.samples))

        self.failUnless((p == clf.predictions).all())
        self.failUnless(N.array(clf.values).shape[0] == N.array(p).shape[0])


def suite():
    return unittest.makeSuite(SMLRTests)


if __name__ == '__main__':
    import runner

