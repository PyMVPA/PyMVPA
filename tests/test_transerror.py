#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

import unittest
import numpy as N
from copy import copy

from mvpa.datasets.dataset import Dataset
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.clfs.transerror import \
     TransferError, ConfusionMatrix, ConfusionBasedError

from mvpa.misc.exceptions import UnknownStateError

from tests_warehouse import normalFeatureDataset, sweepargs
from tests_warehouse_clfs import *

class ErrorsTests(unittest.TestCase):

    def testConfusionMatrix(self):
        data = N.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = N.array([1,1,1,2,2,2,3,3,3])

        cm = ConfusionMatrix()
        self.failUnlessRaises(ZeroDivisionError, lambda x:x.percentCorrect, cm)
        """No samples -- raise exception"""

        cm.add(reg, N.array([1,2,1,2,2,2,3,2,1]))

        self.failUnlessEqual(len(cm.sets), 1,
            msg="Should have a single set so far")
        self.failUnlessEqual(cm.matrix.shape, (3,3),
            msg="should be square matrix (len(reglabels) x len(reglabels)")

        self.failUnlessRaises(ValueError, cm.add, reg, N.array([1]))
        """ConfusionMatrix must complaint if number of samples different"""

        # check table content
        self.failUnless((cm.matrix == [[2,1,0],[0,3,0],[1,1,1]]).all())

        # lets add with new labels (not yet known)
        cm.add(reg, N.array([1,4,1,2,2,2,4,2,1]))

        self.failUnlessEqual(cm.labels, [1,2,3,4],
                             msg="We should have gotten 4th label")

        matrices = cm.matrices          # separate CM per each given set
        self.failUnlessEqual(len(matrices), 2,
                             msg="Have gotten two splits")

        self.failUnless((matrices[0].matrix + matrices[1].matrix == cm.matrix).all(),
                        msg="Total votes should match the sum across split CMs")

        # check pretty print
        # just a silly test to make sure that printing works
        self.failUnless(len(str(cm))>100)
        # and that it knows some parameters for printing
        self.failUnless(len(cm.__str__(summary=True, percents=True,
                                       header=False,
                                       print_empty=True))>100)

        # lets check iadd -- just itself to itself
        cm += cm
        self.failUnlessEqual(len(cm.matrices), 4, msg="Must be 4 sets now")

        # lets check add -- just itself to itself
        cm2 = cm + cm
        self.failUnlessEqual(len(cm2.matrices), 8, msg="Must be 8 sets now")
        self.failUnlessEqual(cm2.percentCorrect, cm.percentCorrect,
                             msg="Percent of corrrect should remain the same ;-)")

        self.failUnlessEqual(cm2.error, 1.0-cm.percentCorrect/100.0,
                             msg="Test if we get proper error value")


    @sweepargs(l_clf=clfs['LinearSVMC'])
    def testConfusionBasedError(self, l_clf):
        train = normalFeatureDataset(perlabel=50, nlabels=2, nfeatures=2,
                                     nonbogus_features=[0,1], snr=3, nchunks=1)
        #l_clf = LinearNuSVMC()
        err = ConfusionBasedError(clf=l_clf)
        terr = TransferError(clf=l_clf)

        self.failUnlessRaises(UnknownStateError, err, None)
        """Shouldn't be able to access the state yet"""

        l_clf.train(train)
        self.failUnlessEqual(err(None), terr(train),
            msg="ConfusionBasedError should be equal to TransferError on" +
                " traindataset")

        # try copying the beast
        terr_copy = copy(terr)


def suite():
    return unittest.makeSuite(ErrorsTests)


if __name__ == '__main__':
    import test_runner

