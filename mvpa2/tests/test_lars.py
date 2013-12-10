# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA least angle regression (LARS) classifier"""

from mvpa2.testing import *
skip_if_no_external('lars')

from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.clfs.lars import LARS
from scipy.stats import pearsonr
from mvpa2.misc.data_generators import normal_feature_dataset

class LARSTests(unittest.TestCase):

    def test_lars(self):
        # not the perfect dataset with which to test, but
        # it will do for now.
        #data = datasets['dumb2']
        # for some reason the R code fails with the dumb data
        data = datasets['chirp_linear']


        clf = LARS()

        clf.train(data)

        # prediction has to be almost perfect
        # test with a correlation
        pre = clf.predict(data.samples)
        cor = pearsonr(pre, data.targets)
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.assertTrue(cor[0] > .8)

    def test_lars_state(self):
        #data = datasets['dumb2']
        # for some reason the R code fails with the dumb data
        data = datasets['chirp_linear']


        clf = LARS()

        clf.train(data)

        clf.ca.enable('predictions')

        p = clf.predict(data.samples)

        self.assertTrue((p == clf.ca.predictions).all())


    def test_lars_sensitivities(self):
        data = datasets['chirp_linear']

        # use LARS on binary problem
        clf = LARS()
        clf.train(data)

        # now ask for the sensitivities WITHOUT having to pass the dataset
        # again
        sens = clf.get_sensitivity_analyzer(force_train=False)(None)

        self.assertTrue(sens.shape == (1, data.nfeatures))


def suite():  # pragma: no cover
    return unittest.makeSuite(LARSTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

