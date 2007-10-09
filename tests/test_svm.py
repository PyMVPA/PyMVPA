### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for SVM classifier
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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import mvpa
import mvpa.svm as svm
import unittest
import numpy


def dumbFeatureSignal():
    data = [[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1],
            [5,0],[5,1],[6,0],[6,1],[7,0],[7,1],[8,0],[8,1],
            [9,0],[9,1],[10,0],[10,1],[11,0],[11,1],[12,0],[12,1]]
    regs = [1 for i in range(8)] \
         + [2 for i in range(8)] \
         + [3 for i in range(8)]

    return mvpa.MVPAPattern(data, regs)


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
    data=numpy.random.normal(size=(4*patterns,2))

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
    regs = numpy.array(regs)

    return mvpa.MVPAPattern(data, regs)


class SVMTests(unittest.TestCase):

    def testCapabilityReport(self):
        clf = svm.SVM()
        clf.train(dumbFeatureSignal())
        self.failUnless('feature_benchmark' in clf.capabilities)


    def testMultivariate(self):

        mv_perf = []
        uv_perf = []

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            s_mv = svm.SVM()
            s_mv.train(train)
            p_mv = s_mv.predict( test.pattern )
            mv_perf.append( numpy.mean(p_mv==test.reg) )

            s_uv = svm.SVM()
            s_uv.train(train.selectFeaturesById([0]))
            p_uv = s_uv.predict( test.selectFeaturesById([0]).pattern )
            uv_perf.append( numpy.mean(p_uv==test.reg) )

        mean_mv_perf = numpy.mean(mv_perf)
        mean_uv_perf = numpy.mean(uv_perf)

        self.failUnless( mean_mv_perf > 0.9 )
        self.failUnless( mean_uv_perf < mean_mv_perf )


    def testFeatureBenchmark(self):
        pat = dumbFeatureSignal()
        clf = svm.SVM()
        clf.train(pat)
        rank = clf.getFeatureBenchmark()

        # has to be 1d array
        self.failUnless( len(rank.shape) == 1 ) 

        # has to be one value per feature
        self.failUnless( len(rank) == pat.nfeatures )

        # first feature is discriminative, second is not
        self.failUnless(rank[0] > rank[1])


def suite():
    return unittest.makeSuite(SVMTests)


if __name__ == '__main__':
    unittest.main()

