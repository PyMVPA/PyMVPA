### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA logistic regression classifier
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import unittest
import mvpa
import mvpa.plf as plf
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

    return mvpa.Dataset(data, regs, None)


class PLFTests(unittest.TestCase):


    def testMultivariate(self):

        mv_perf = []
        uv_perf = []

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            k_mv = plf.PLF()
            k_mv.train(train)
            p_mv = k_mv.predict( test.samples )
            mv_perf.append( np.mean(p_mv==test.labels) )

            k_uv = plf.PLF()
            k_uv.train(train.selectFeatures([0]))
            p_uv = k_uv.predict( test.selectFeatures([0]).samples )
            uv_perf.append( np.mean(p_uv==test.labels) )

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

