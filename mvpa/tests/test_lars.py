# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA least angle regression (LARS) classifier"""

from mvpa import cfg
from mvpa.clfs.lars import LARS
from scipy.stats import pearsonr
from tests_warehouse import *
from mvpa.misc.data_generators import normalFeatureDataset

class LARSTests(unittest.TestCase):

    def testLARS(self):
        # not the perfect dataset with which to test, but
        # it will do for now.
        #data = datasets['dumb2']
        # for some reason the R code fails with the dumb data
        data = datasets['chirp_linear']


        clf = LARS(regression=True)

        clf.train(data)

        # prediction has to be almost perfect
        # test with a correlation
        pre = clf.predict(data.samples)
        cor = pearsonr(pre, data.labels)
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(cor[0] > .8)

    def testLARSState(self):
        #data = datasets['dumb2']
        # for some reason the R code fails with the dumb data
        data = datasets['chirp_linear']


        clf = LARS()

        clf.train(data)

        clf.states.enable('predictions')

        p = clf.predict(data.samples)

        self.failUnless((p == clf.predictions).all())


    def testLARSSensitivities(self):
        data = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=4)

        # use LARS on binary problem
        clf = LARS()
        clf.train(data)

        # now ask for the sensitivities WITHOUT having to pass the dataset
        # again
        sens = clf.getSensitivityAnalyzer(force_training=False)()

        self.failUnless(sens.shape == (data.nfeatures,))


def suite():
    return unittest.makeSuite(LARSTests)


if __name__ == '__main__':
    import runner

