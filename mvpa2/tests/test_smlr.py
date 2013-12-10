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


class SMLRTests(unittest.TestCase):

    def test_smlr(self):
        data = datasets['dumb']

        clf = SMLR()

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
        self.assertTrue((predictions == data.targets).all())


    def test_smlr_state(self):
        data = datasets['dumb']

        clf = SMLR()

        clf.train(data)

        clf.ca.enable('estimates')
        clf.ca.enable('predictions')

        p = np.asarray(clf.predict(data.samples))

        self.assertTrue((p == clf.ca.predictions).all())
        self.assertTrue(np.array(clf.ca.estimates).shape[0] == np.array(p).shape[0])


    def test_smlr_sensitivities(self):
        data = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)

        # use SMLR on binary problem, but not fitting all weights
        clf = SMLR(fit_all_weights=False)
        clf.train(data)

        # now ask for the sensitivities WITHOUT having to pass the dataset
        # again
        sens = clf.get_sensitivity_analyzer(force_train=False)(None)
        self.assertTrue(sens.shape == (len(data.UT) - 1, data.nfeatures))


def suite():  # pragma: no cover
    return unittest.makeSuite(SMLRTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

