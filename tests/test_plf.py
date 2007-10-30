#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA logistic regression classifier"""

import unittest
from mvpa.datasets.dataset import Dataset
from mvpa.clf.plf import PLF
import numpy as N


def pureMultivariateSignal(patterns, signal2noise = 1.5):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    %%%%%%%%%
    % O % X %
    %%%%%%%%%
    % X % O %
    %%%%%%%%%
    """

    # start with noise
    data=N.random.normal(size=(4*patterns,2))

    # add signal
    data[:2*patterns,1] += signal2noise
    data[2*patterns:4*patterns,1] -= signal2noise
    data[:patterns,0] -= signal2noise
    data[2*patterns:3*patterns,0] -= signal2noise
    data[patterns:2+patterns,0] += signal2noise
    data[3*patterns:4*patterns,0] += signal2noise

    # two conditions
    regs = [0 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [0 for i in xrange(patterns)]
    regs = N.array(regs)

    return Dataset(data, regs, None)


class PLFTests(unittest.TestCase):


    def testMultivariate(self):

        mv_perf = []
        uv_perf = []

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            k_mv = PLF()
            k_mv.train(train)
            p_mv = k_mv.predict( test.samples )
            mv_perf.append( N.mean(p_mv==test.labels) )

            k_uv = PLF()
            k_uv.train(train.selectFeatures([0]))
            p_uv = k_uv.predict( test.selectFeatures([0]).samples )
            uv_perf.append( N.mean(p_uv==test.labels) )

        mean_mv_perf = N.mean(mv_perf)
        mean_uv_perf = N.mean(uv_perf)

        print mean_mv_perf
        print mean_uv_perf

        # multivariate should be perfect
        self.failUnless( mean_mv_perf > 0.9 )
        # univariate should be worse
        self.failUnless( mean_uv_perf < mean_mv_perf )


def suite():
    return unittest.makeSuite(PLFTests)


if __name__ == '__main__':
    unittest.main()

