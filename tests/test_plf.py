### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA logistic regression classifier
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import unittest
import mvpa
import mvpa.plf as plf
import numpy as np


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
    data=np.random.normal(size=(4*patterns,2))

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
    regs = np.array(regs)

    return mvpa.MVPAPattern(data, regs)


class PLFTests(unittest.TestCase):


    def testMultivariate(self):

        mv_perf = []
        uv_perf = []

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            k_mv = plf.PLF(train)
            p_mv = k_mv.predict( test.pattern )
            mv_perf.append( np.mean(p_mv==test.reg) )

            k_uv = plf.PLF(train.selectFeatures([0]))
            p_uv = k_uv.predict( test.selectFeatures([0]).pattern )
            uv_perf.append( np.mean(p_uv==test.reg) )

        mean_mv_perf = np.mean(mv_perf)
        mean_uv_perf = np.mean(uv_perf)

        self.failUnless( mean_mv_perf > 0.9 )
        self.failUnless( mean_uv_perf < mean_mv_perf )


def suite():
    return unittest.makeSuite(PLFTests)


if __name__ == '__main__':
    unittest.main()

