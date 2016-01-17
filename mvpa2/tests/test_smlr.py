# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA sparse multinomial logistic regression classifier"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2.clfs.smlr import SMLR
from mvpa2.misc.data_generators import normal_feature_dataset


@sweepargs(clf=(SMLR(), SMLR(implementation='Python')))
def test_smlr(clf):
    data = datasets['dumb']

    clf.train(data)

    # prediction has to be perfect
    #
    # XXX yoh: whos said that?? ;-)
    #
    # There is always a tradeoff between learning and
    # generalization errors so...  but in this case the problem is
    # more interesting: absent bias disallows to learn data you
    # have here -- there is no solution which would pass through
    # (0,0)
    predictions = clf.predict(data.samples)
    assert_array_equal(predictions, data.targets)


def test_smlr_state():
    data = datasets['dumb']

    clf = SMLR()

    clf.train(data)

    clf.ca.enable('estimates')
    clf.ca.enable('predictions')

    p = np.asarray(clf.predict(data.samples))

    assert_array_equal(p, clf.ca.predictions)
    assert_equal(np.array(clf.ca.estimates).shape[0], np.array(p).shape[0])


@sweepargs(clf=(SMLR(fit_all_weights=False),
                SMLR(fit_all_weights=False, unsparsify=True)))
def test_smlr_sensitivities(clf):
    data = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)

    # use SMLR on binary problem, but not fitting all weights
    clf.train(data)

    # now ask for the sensitivities WITHOUT having to pass the dataset
    # again
    sens = clf.get_sensitivity_analyzer(force_train=False)(None)
    assert_equal(sens.shape, (len(data.UT) - 1, data.nfeatures))
