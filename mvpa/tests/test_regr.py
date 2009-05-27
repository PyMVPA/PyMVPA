# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Regressions"""

from mvpa.base import externals
from mvpa.support.copy import deepcopy

from mvpa.datasets import Dataset
from mvpa.mappers.mask import MaskMapper
from mvpa.datasets.splitters import NFoldSplitter, OddEvenSplitter

from mvpa.misc.errorfx import RMSErrorFx, RelativeRMSErrorFx, \
     CorrErrorFx, CorrErrorPFx

from mvpa.clfs.meta import SplitClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.misc.exceptions import UnknownStateError

from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from tests_warehouse import *
from tests_warehouse_clfs import *

class RegressionsTests(unittest.TestCase):

    @sweepargs(ml=clfswh['regression']+regrswh[:])
    def testNonRegressions(self, ml):
        """Test If binary regression-based  classifiers have proper tag
        """
        self.failUnless(('binary' in ml._clf_internals) != ml.regression,
                        msg="Inconsistent markin with "
                        "binary and regression features detected")

    @sweepargs(regr=regrswh['regression'])
    def testRegressions(self, regr):
        """Simple tests on regressions
        """
        ds = datasets['chirp_linear']

        cve = CrossValidatedTransferError(
            TransferError(regr, CorrErrorFx()),
            splitter=NFoldSplitter(),
            enable_states=['training_confusion', 'confusion'])
        corr = cve(ds)

        splitregr = SplitClassifier(regr,
                                    splitter=OddEvenSplitter(),
                                    enable_states=['training_confusion', 'confusion'])
        splitregr.train(ds)
        split_corr = cve.confusion.stats['Summary CCe']

        for confusion, error in ((cve.confusion, corr),
                                 (splitregr.confusion, split_corr),
                                 (splitregr.training_confusion, split_corr),
                                 ):
            #TODO: test confusion statistics
            s0 = confusion.asstring(short=True)
            s1 = confusion.asstring(short=False)

            for s in [s0, s1]:
                self.failUnless(len(s) > 10,
                                msg="We should get some string representation "
                                "of regression summary. Got %s" % s)

            self.failUnless(error<0.2,
                            msg="Regressions should perform well on a simple "
                            "dataset. Got correlation error of %s " % error)

            # Test access to summary statistics
            if cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnless(confusion.stats['Summary CCe'] < 0.5)

        split_predictions = splitregr.predict(ds.samples) # just to check if it works fine

        # To test basic plotting
        #import pylab as P
        #cve.confusion.plot()
        #P.show()

    @sweepargs(clf=clfswh['regression'])
    def testRegressionsClassifiers(self, clf):
        """Simple tests on regressions being used as classifiers
        """
        # check if we get values set correctly
        clf.states._changeTemporarily(enable_states=['values'])
        self.failUnlessRaises(UnknownStateError, clf.states['values']._get)
        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        ds = datasets['uni2small']
        cverror = cv(ds)
        self.failUnless(len(clf.values) == ds['chunks', 1].nsamples)
        clf.states._resetEnabledTemporarily()



def suite():
    return unittest.makeSuite(RegressionsTests)


if __name__ == '__main__':
    import runner
