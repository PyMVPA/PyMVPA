# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA least angle regression (ENET) classifier"""

from mvpa.testing.tools import assert_true, assert_equal
from mvpa.testing.tools import assert_array_equal

from mvpa import cfg
from mvpa.clfs.glmnet import GLMNET_R,GLMNET_C

#from scipy.stats import pearsonr
# Lets use our CorrErrorFx which would be available even without scipy
from mvpa.misc.errorfx import CorrErrorFx
from tests_warehouse import *
from mvpa.misc.data_generators import normalFeatureDataset

def test_glmnet_r():
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
    corerr = CorrErrorFx()(pre, data.labels)
    if cfg.getboolean('tests', 'labile', default='yes'):
        assert_true(corerr < .2)

def test_glmnet_c():
    # define binary prob
    data = datasets['dumb2']

    # use GLMNET on binary problem
    clf = GLMNET_C()
    clf.states.enable('estimates')

    clf.train(data)

    # test predictions
    pre = clf.predict(data.samples)

    assert_array_equal(pre, data.labels)

def test_glmnet_state():
    #data = datasets['dumb2']
    # for some reason the R code fails with the dumb data
    data = datasets['chirp_linear']

    clf = GLMNET_R()

    clf.train(data)

    clf.states.enable('predictions')

    p = clf.predict(data.samples)

    assert_array_equal(p, clf.states.predictions)


def test_glmnet_c_sensitivities():
    data = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=4)

    # use GLMNET on binary problem
    clf = GLMNET_C()
    clf.train(data)

    # now ask for the sensitivities WITHOUT having to pass the dataset
    # again
    sens = clf.getSensitivityAnalyzer(force_training=False)()

    #failUnless(sens.shape == (data.nfeatures,))
    assert_equal(sens.shape, (len(data.UL), data.nfeatures))

def test_glmnet_r_sensitivities():
    data = datasets['chirp_linear']

    clf = GLMNET_R()

    clf.train(data)

    # now ask for the sensitivities WITHOUT having to pass the dataset
    # again
    sens = clf.getSensitivityAnalyzer(force_training=False)()

    assert_equal(sens.shape, (data.nfeatures,))
