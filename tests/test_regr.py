#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Regressions"""

from mvpa.misc.copy import deepcopy

from mvpa.datasets import Dataset
from mvpa.mappers import MaskMapper
from mvpa.datasets.splitter import NFoldSplitter

from mvpa.misc.errorfx import CorrErrorFx, RMSErrorFx

from mvpa.clfs.transerror import TransferError
from mvpa.misc.exceptions import UnknownStateError

from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from tests_warehouse import *
from tests_warehouse_clfs import *

class RegressionsTests(unittest.TestCase):

    @sweepargs(ml=clfs['regression']+regrs[:])
    def testNonRegressions(self, ml):
        """Test If binary regression-based  classifiers have proper tag
        """
        self.failUnless(('binary' in ml._clf_internals) != ml.regression,
                        msg="Inconsistent markin with "
                        "binary and regression features detected")

    @sweepargs(regr=regrs['regression'])
    def testRegressions(self, regr):
        """Simple tests on regressions
        """
        ds = datasets['chirp_linear']

        cve = CrossValidatedTransferError(
            TransferError(regr, CorrErrorFx()),
            splitter=NFoldSplitter(),
            enable_states=['training_confusion'])
        corr = cve(ds)
        self.failUnless(corr>0.9,
                        msg="Regressions should perform well on a simple "
                        "dataset. Got mean correlation of %s " % corr)
        #TODO: test confusion statistics
        #print cve.training_confusion

def suite():
    return unittest.makeSuite(RegressionsTests)


if __name__ == '__main__':
    import runner
