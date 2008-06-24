#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Regressions"""

from mvpa.base import externals
from mvpa.misc.copy import deepcopy

from mvpa.datasets import Dataset
from mvpa.mappers import MaskMapper
from mvpa.datasets.splitter import NFoldSplitter

from mvpa.misc.errorfx import RMSErrorFx, RelativeRMSErrorFx, \
     CorrErrorFx, CorrErrorPFx

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
            enable_states=['training_confusion', 'confusion'])
        corr = cve(ds)

        #TODO: test confusion statistics
        s0 = cve.confusion.asstring(short=True)
        s1 = cve.confusion.asstring(short=False)

        for s in [s0, s1]:
            self.failUnless(len(s) > 10,
                            msg="We should get some string representation "
                            "of regression summary. Got %s" % s)

        self.failUnless(corr<0.2,
                        msg="Regressions should perform well on a simple "
                        "dataset. Got correlation error of %s " % corr)


def suite():
    return unittest.makeSuite(RegressionsTests)


if __name__ == '__main__':
    import runner
