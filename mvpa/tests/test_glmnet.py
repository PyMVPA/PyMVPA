# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA least angle regression (ENET) classifier"""

from mvpa import cfg
from mvpa.clfs.glmnet import GLMNET_R,GLMNET_C
from scipy.stats import pearsonr
from tests_warehouse import *
from mvpa.misc.data_generators import normalFeatureDataset

class GLMNETTests(unittest.TestCase):

    def testGLMNET_R(self):
        # not the perfect dataset with which to test, but
        # it will do for now.
        #data = datasets['dumb2']
        # for some reason the R code fails with the dumb data
        data = datasets['chirp_linear']

        clf = GLMNET_R()

        clf.train(data)

        # prediction has to be almost perfect
        # test with a correlation
        pre = clf.predict(data.samples)
        cor = pearsonr(pre, data.labels)
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(cor[0] > .8)

    def testGLMNET_C(self):
        # define binary prob
        data = datasets['dumb2']

        # use GLMNET on binary problem
        clf = GLMNET_C()
        clf.states.enable('estimates')

        clf.train(data)

        # test predictions
        pre = clf.predict(data.samples)

        self.failUnless((pre == data.labels).all())

    def testGLMNETState(self):
        #data = datasets['dumb2']
        # for some reason the R code fails with the dumb data
        data = datasets['chirp_linear']

        clf = GLMNET_R()

        clf.train(data)

        clf.states.enable('predictions')

        p = clf.predict(data.samples)

        self.failUnless((p == clf.states.predictions).all())


    def testGLMNET_CSensitivities(self):
        data = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=4)

        # use GLMNET on binary problem
        clf = GLMNET_C()
        clf.train(data)

        # now ask for the sensitivities WITHOUT having to pass the dataset
        # again
        sens = clf.getSensitivityAnalyzer(force_training=False)()

        #self.failUnless(sens.shape == (data.nfeatures,))
        self.failUnless(sens.shape == (len(data.UL), data.nfeatures))

    def testGLMNET_RSensitivities(self):
        data = datasets['chirp_linear']

        clf = GLMNET_R()

        clf.train(data)

        # now ask for the sensitivities WITHOUT having to pass the dataset
        # again
        sens = clf.getSensitivityAnalyzer(force_training=False)()

        self.failUnless(sens.shape == (data.nfeatures,))


def suite():
    return unittest.makeSuite(ENETTests)


if __name__ == '__main__':
    import runner

