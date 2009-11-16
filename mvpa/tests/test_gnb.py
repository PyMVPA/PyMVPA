# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GNB classifier"""

from mvpa.clfs.gnb import GNB
from tests_warehouse import *

class GNBTests(unittest.TestCase):

    def testGNB(self):
        gnb = GNB()
        gnb_nc = GNB(common_variance=False)
        gnb_n = GNB(normalize=True)
        gnb_n_nc = GNB(normalize=True, common_variance=False)

        ds_tr = datasets['uni2medium_train']
        ds_te = datasets['uni2medium_test']

        # Generic silly coverage just to assure that it works in all
        # possible scenarios:
        bools = (True, False)
        # There should be better way... heh
        for cv in bools:                # common_variance?
          for up in bools:
            tp = None                   # predictions -- all above should
                                        # result in the same predictions
            for n in bools:             # normalized?
              for ls in bools:          # logspace?
                for es in ((), ('values')):
                    gnb_ = GNB(common_variance=cv,
                               uniform_prior=up,
                               normalize=n,
                               logprob=ls,
                               enable_states=es)
                    gnb_.train(ds_tr)
                    predictions = gnb_.predict(ds_te.samples)
                    if tp is None:
                        tp = predictions
                    self.failUnless((predictions == tp),
                                    msg="%s failed to reproduce predictions" %
                                    gnb_)
                    # if normalized -- check if values are such
                    if n and 'values' in es:
                        v = gnb_.values
                        if ls:          # in log space -- take exp ;)
                            v = N.exp(v)
                        d1 = N.sum(v, axis=1) - 1.0
                        self.failUnless(N.max(N.abs(d1)) < 1e-5)

def suite():
    return unittest.makeSuite(GNBTests)


if __name__ == '__main__':
    import runner

