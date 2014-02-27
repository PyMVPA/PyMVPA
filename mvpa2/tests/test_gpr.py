# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GPR."""

from mvpa2.base import externals
from mvpa2.misc import data_generators
from mvpa2.misc.attrmap import AttributeMap
from mvpa2.kernels.np import GeneralizedLinearKernel
from mvpa2.clfs.gpr import GPR

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets
from mvpa2.testing.tools import assert_array_equal, assert_array_almost_equal

if __debug__:
    from mvpa2.base import debug


class GPRTests(unittest.TestCase):

    def test_basic(self):
        skip_if_no_external('scipy') # needed by GPR code
        dataset = data_generators.linear1d_gaussian_noise()
        k = GeneralizedLinearKernel()
        clf = GPR(k)
        clf.train(dataset)
        y = clf.predict(dataset.samples)
        assert_array_equal(y.shape, dataset.targets.shape)

    def test_linear(self):
        pass

    def _test_gpr_model_selection(self):  # pragma: no cover
        """Smoke test for running model selection while getting GPRWeights

        TODO: DISABLED because setting of hyperparameters was not adopted for 0.6 (yet)
        """
        if not externals.exists('openopt'):
            return
        amap = AttributeMap()           # we would need to pass numbers into the GPR
        dataset = datasets['uni2small'].copy() #data_generators.linear1d_gaussian_noise()
        dataset.targets = amap.to_numeric(dataset.targets).astype(float)
        k = GeneralizedLinearKernel()
        clf = GPR(k, enable_ca=['log_marginal_likelihood'])
        sa = clf.get_sensitivity_analyzer() # should be regular weights
        sa_ms = clf.get_sensitivity_analyzer(flavor='model_select') # with model selection
        def prints():
            print clf.ca.log_marginal_likelihood, clf.kernel.Sigma_p, clf.kernel.sigma_0

        sa(dataset)
        lml = clf.ca.log_marginal_likelihood

        sa_ms(dataset)
        lml_ms = clf.ca.log_marginal_likelihood

        self.assertTrue(lml_ms > lml)



def suite():  # pragma: no cover
    return unittest.makeSuite(GPRTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
