# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GPR."""

import unittest

from mvpa.base import externals

from mvpa.misc import data_generators
from mvpa.clfs.kernel import KernelLinear as GeneralizedLinearKernel
from mvpa.clfs.gpr import GPR

from tests_warehouse import *
from numpy.testing import assert_array_equal, assert_array_almost_equal

if __debug__:
    from mvpa.base import debug
    

class GPRTests(unittest.TestCase):

    def test_basic(self):
        dataset = data_generators.linear1d_gaussian_noise()
        k = GeneralizedLinearKernel()
        clf = GPR(k)
        clf.train(dataset)
        y = clf.predict(dataset.samples)
        assert_array_equal(y.shape, dataset.labels.shape)

    def test_linear(self):
        pass

    def test_gpr_model_selection(self):
        """Smoke test for running model selection while getting GPRWeights
        """
        if not externals.exists('openopt'):
            return

        dataset = datasets['uni2small'] #data_generators.linear1d_gaussian_noise()
        k = GeneralizedLinearKernel()
        clf = GPR(k, enable_states=['log_marginal_likelihood'])
        sa = clf.getSensitivityAnalyzer() # should be regular weights
        sa_ms = clf.getSensitivityAnalyzer(flavor='model_select') # with model selection
        def prints():
            print clf.states.log_marginal_likelihood, clf.kernel.Sigma_p, clf.kernel.sigma_0

        sa(dataset)
        lml = clf.states.log_marginal_likelihood

        sa_ms(dataset)
        lml_ms = clf.states.log_marginal_likelihood

        self.failUnless(lml_ms > lml)



def suite():
    return unittest.makeSuite(GPRTests)


if __name__ == '__main__':
    import runner
