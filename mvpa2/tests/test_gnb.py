# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GNB classifier"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.datasets import *

from mvpa2.clfs.gnb import GNB
from mvpa2.measures.base import TransferMeasure
from mvpa2.generators.splitters import Splitter

class GNBTests(unittest.TestCase):

    def test_gnb(self):
        gnb = GNB()
        gnb_nc = GNB(common_variance=False)
        gnb_n = GNB(normalize=True)
        gnb_n_nc = GNB(normalize=True, common_variance=False)

        ds = datasets['uni2medium']

        # Generic silly coverage just to assure that it works in all
        # possible scenarios:
        bools = (True, False)
        # There should be better way... heh
        for cv in bools:                # common_variance?
          for prior in ('uniform', 'laplacian_smoothing', 'ratio'):
            tp = None                   # predictions -- all above should
                                        # result in the same predictions
            for n in bools:             # normalized?
              for ls in bools:          # logspace?
                for es in ((), ('estimates')):
                    gnb_ = GNB(common_variance=cv,
                               prior=prior,
                               normalize=n,
                               logprob=ls,
                               enable_ca=es)
                    tm = TransferMeasure(gnb_, Splitter('train'))
                    predictions = tm(ds).samples[:,0]
                    if tp is None:
                        tp = predictions
                    assert_array_equal(predictions, tp)
                    # if normalized -- check if estimates are such
                    if n and 'estimates' in es:
                        v = gnb_.ca.estimates
                        if ls:          # in log space -- take exp ;)
                            v = np.exp(v)
                        d1 = np.sum(v, axis=1) - 1.0
                        self.assertTrue(np.max(np.abs(d1)) < 1e-5)

def suite():  # pragma: no cover
    return unittest.makeSuite(GNBTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

