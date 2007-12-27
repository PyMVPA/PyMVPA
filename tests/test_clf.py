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
from mvpa.datasets.maskmapper import MaskMapper
from mvpa.datasets.splitter import NFoldSplitter

from mvpa.clfs.classifier import Classifier, CombinedClassifier, \
     BinaryClassifier, MulticlassClassifier, \
     SplitClassifier, MappedClassifier, FeatureSelectionClassifier

from copy import deepcopy

class SameSignClassifier(Classifier):
    """Dummy classifier which reports +1 class if both features have
    the same sign, -1 otherwise"""

    def __init__(self):
        Classifier.__init__(self)
    def _train(self, data):
        # we don't need that ;-)
        pass
    def predict(self, data):
        datalen = len(data)
        values = []
        for d in data:
            values.append(2*int( (d[0]>=0) == (d[1]>=0) )-1)
        self["predictions"] = values
        return values


class Less1Classifier(SameSignClassifier):
    """Dummy classifier which reports +1 class if abs value of max less than 1"""
    def predict(self, data):
        datalen = len(data)
        values = []
        for d in data:
            values.append(2*int(max(d)<=1)-1)
        self["predictions"] = values
        return values


class ClassifiersTests(unittest.TestCase):

    def setUp(self):
        self.clf_sign = SameSignClassifier()
        self.clf_less1 = Less1Classifier()

        # simple binary dataset
        self.data_bin_1 = Dataset(
            samples=[[0,0],[-10,-1],[1,0.1],[1,-1],[-1,1]],
            labels=[1, 1, 1, -1, -1], # labels
            chunks=[0, 1, 2,  2, 3])  # chunks

    def testDummy(self):
        clf = SameSignClassifier()
        clf.train(self.data_bin_1)
        self.failUnlessEqual(clf["trained_confusion"].percentCorrect,
                             100,
                             msg="Dummy clf should train perfectly")
        self.failUnlessEqual(clf.predict(self.data_bin_1.samples),
                             list(self.data_bin_1.labels))

    def testBoosted(self):
        # XXXXXXX
        # silly test if we get the same result with boosted as with a single one
        bclf = CombinedClassifier(clfs=[deepcopy(self.clf_sign),
                                        deepcopy(self.clf_sign)])
        self.failUnlessEqual(list(bclf.predict(self.data_bin_1.samples)),
                             list(self.data_bin_1.labels),
                             msg="Boosted classifier should work")
        self.failUnlessEqual(bclf.predict(self.data_bin_1.samples),
                             self.clf_sign.predict(self.data_bin_1.samples),
                             msg="Boosted classifier should have the same as regular")


    def testBinaryDecorator(self):
        ds = Dataset(samples=[ [0,0], [0,1], [1,100], [-1,0], [-1,-3], [ 0,-10] ],
                     labels=[ 'sp', 'sp', 'sp', 'dn', 'sn', 'dp'])
        testdata = [ [0,0], [10,10], [-10, -1], [0.1, -0.1], [-0.2, 0.2] ]
        # labels [s]ame/[d]ifferent (sign), and [p]ositive/[n]egative first element

        clf = SameSignClassifier()
        # lets create classifier to descriminate only between same/different,
        # which is a primary task of SameSignClassifier
        bclf1 = BinaryClassifier(clf=clf,
                                 poslabels=['sp', 'sn'],
                                 neglabels=['dp', 'dn'])
        self.failUnless(bclf1.predict(testdata) ==
                        [['sp', 'sn'], ['sp', 'sn'], ['sp', 'sn'],
                         ['dn', 'dp'], ['dn', 'dp']])

        # check by selecting just 
        #self. fail


    def testSplitClassifier(self):
        ds = self.data_bin_1
        clf = SplitClassifier(clf=SameSignClassifier(),
                              splitter=NFoldSplitter(1))
        clf.train(ds)                   # train the beast
        self.failUnlessEqual(clf["trained_confusions"].percentCorrect,
                             100,
                             msg="Dummy clf should train perfectly")
        self.failUnlessEqual(len(clf["trained_confusions"].sets),
                             len(ds.uniquechunks),
                             msg="Should have 1 confusion per each split")
        self.failUnlessEqual(len(clf.clfs), len(ds.uniquechunks),
                             msg="Should have number of classifiers equal # of epochs")
        self.failUnlessEqual(clf.predict(ds.samples), list(ds.labels),
                             msg="Should classify correctly")


    def testMappedClassifier(self):
        samples = N.array([ [0,0,-1], [1,0,1], [-1,-1, 1], [-1,0,1], [1, -1, 1] ])
        testdata3 = Dataset(samples=samples, labels=1)
        res110 = [1, 1, 1, -1, -1]
        res101 = [-1, 1, -1, -1, 1]
        res011 = [-1, 1, -1, 1, -1]

        clf110 = MappedClassifier(clf=self.clf_sign, mapper=MaskMapper(N.array([1,1,0])))
        clf101 = MappedClassifier(clf=self.clf_sign, mapper=MaskMapper(N.array([1,0,1])))
        clf011 = MappedClassifier(clf=self.clf_sign, mapper=MaskMapper(N.array([0,1,1])))

        self.failUnlessEqual(clf110.predict(samples), res110)
        self.failUnlessEqual(clf101.predict(samples), res101)
        self.failUnlessEqual(clf011.predict(samples), res011)


    def testFeatureSelectionClassifier(self):
        from test_rfe import SillySensitivityAnalyzer
        from mvpa.algorithms.featsel import \
             SensitivityBasedFeatureSelection, \
             FixedNElementTailSelector

        # should give lowest weight to the feature with lowest index
        sens_ana = SillySensitivityAnalyzer()
        # should give lowest weight to the feature with highest index
        sens_ana_rev = SillySensitivityAnalyzer(mult=-1)

        # corresponding feature selections
        feat_sel = SensitivityBasedFeatureSelection(sens_ana,
            FixedNElementTailSelector(1))

        feat_sel_rev = SensitivityBasedFeatureSelection(sens_ana_rev,
            FixedNElementTailSelector(1))

        samples = N.array([ [0,0,-1], [1,0,1], [-1,-1, 1], [-1,0,1], [1, -1, 1] ])

        testdata3 = Dataset(samples=samples, labels=1)
        # dummy train data so proper mapper gets created
        traindata = Dataset(samples=N.array([ [0, 0,-1], [1,0,1] ]), labels=[1,2])

        # targets
        res110 = [1, 1, 1, -1, -1]
        res011 = [-1, 1, -1, 1, -1]

        # first classifier -- 0th feature should be discarded
        clf011 = FeatureSelectionClassifier(self.clf_sign, feat_sel)
        clf011.train(traindata)
        self.failUnlessEqual(clf011.predict(testdata3.samples), res011)

        # first classifier -- last feature should be discarded
        clf011 = FeatureSelectionClassifier(self.clf_sign, feat_sel_rev)
        clf011.train(traindata)
        self.failUnlessEqual(clf011.predict(testdata3.samples), res110)


def suite():
    return unittest.makeSuite(ClassifiersTests)


if __name__ == '__main__':
    import test_runner
