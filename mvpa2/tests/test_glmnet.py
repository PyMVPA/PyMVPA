# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA least angle regression (ENET) classifier"""

from mvpa2.testing import *
skip_if_no_external('glmnet')

from mvpa2.testing.datasets import *

from mvpa2 import cfg
from mvpa2.clfs.glmnet import GLMNET_R,GLMNET_C

#from scipy.stats import pearsonr
# Lets use our corr_error which would be available even without scipy
from mvpa2.misc.errorfx import corr_error
from mvpa2.misc.data_generators import normal_feature_dataset

from mvpa2.testing.tools import assert_true, assert_equal, assert_array_equal

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
    corerr = corr_error(pre, data.targets)
    if cfg.getboolean('tests', 'labile', default='yes'):
        assert_true(corerr < .2)

def test_glmnet_c():
    # define binary prob
    data = datasets['dumb2']

    # use GLMNET on binary problem
    clf = GLMNET_C()
    clf.ca.enable('estimates')

    clf.train(data)

    # test predictions
    pre = clf.predict(data.samples)

    assert_array_equal(pre, data.targets)

def test_glmnet_state():
    #data = datasets['dumb2']
    # for some reason the R code fails with the dumb data
    data = datasets['chirp_linear']

    clf = GLMNET_R()

    clf.train(data)

    clf.ca.enable('predictions')

    p = clf.predict(data.samples)

    assert_array_equal(p, clf.ca.predictions)


def test_glmnet_c_sensitivities():
    data = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)

    # use GLMNET on binary problem
    clf = GLMNET_C()
    clf.train(data)

    # now ask for the sensitivities WITHOUT having to pass the dataset
    # again
    sens = clf.get_sensitivity_analyzer(force_train=False)(None)

    #failUnless(sens.shape == (data.nfeatures,))
    assert_equal(sens.shape, (len(data.UT), data.nfeatures))

def test_glmnet_r_sensitivities():
    data = datasets['chirp_linear']

    clf = GLMNET_R()

    clf.train(data)

    # now ask for the sensitivities WITHOUT having to pass the dataset
    # again
    sens = clf.get_sensitivity_analyzer(force_train=False)(None)

    assert_equal(sens.shape, (1, data.nfeatures))
