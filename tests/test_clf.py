#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA basic Classifiers"""

import unittest

import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.clf.classifier import Classifier, BoostedClassifier, \
     BinaryClassifierDecorator, BoostedMulticlassClassifier


class DummyClassifier(Classifier):
    """Dummy classifier which reports +1 class if both features have
    the same sign, -1 otherwise"""

    def __init__(self):
        Classifier.__init__(self)
    def train(self, data):
        # we don't need that ;-)
        pass
    def predict(self, data):
        datalen = len(data)
        values = []
        for d in data:
            values.append(2*int(d[0]*d[1]>=0)-1)
        return values


class ClassifiersTests(unittest.TestCase):

    def testDummy(self):
        clf = DummyClassifier()
        clf.train(None)
        self.failUnless(clf.predict([[0,0],[-10,-1],[1,0.1],[1,-1],[-1,1]])
                        == [1, 1, 1, -1, -1])

    def testBoosted(self):
        # XXXXXXX
        pass

    def testBinaryDecorator(self):
        ds = Dataset(samples=[ [0,0], [0,1], [1,100], [-1,0], [-1,-3], [ 0,-10] ],
                     labels=[ 'sp', 'sp', 'sp', 'dn', 'sn', 'dp'])
        testdata = [ [0,0], [10,10], [-10, -1], [0.1, -0.1], [-0.2, 0.2] ]
        # labels [s]ame/[d]ifferent (sign), and [p]ositive/[n]egative first element

        clf = DummyClassifier()
        # lets create classifier to descriminate only between same/different,
        # which is a primary task of DummyClassifier
        bclf1 = BinaryClassifierDecorator(clf=clf,
                                          poslabels=['sp', 'sn'],
                                          neglabels=['dp', 'dn'])
        self.failUnless(bclf1.predict(testdata) ==
                        [['sp', 'sn'], ['sp', 'sn'], ['sp', 'sn'],
                         ['dn', 'dp'], ['dn', 'dp']])

        # check by selecting just 
        #self. fail

def suite():
    return unittest.makeSuite(ClassifiersTests)


if __name__ == '__main__':
    import test_runner

