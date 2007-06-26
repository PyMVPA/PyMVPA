### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA serial feature inclusion algorithm
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

import mvpa.serialinclusion as si
import mvpa
import mvpa.knn as knn
import mvpa.svm_wrap as svm_wrap
import unittest
import numpy as np
import scipy.stats as stats

class SerialInclusionTests(unittest.TestCase):

    def testSerialInclusion(self):
        # prepare demo dataset and mask
        mask = np.ones((3,3,3))
        data = np.random.uniform(0,1,(100,) + (mask.shape))
        reg = np.arange(100) / 50
        orig = range(10) * 10
        pattern = mvpa.MVPAPattern(data, reg, orig)

        # init algorithm
        sinc = si.SerialFeatureInclusion( pattern,
                                          mask,
                                          ncvfoldsamples=1 )

        # run selection of single features
        sinc.selectFeatures( svm_wrap.SVM, verbose=True)

        # no real check yet, simply checking something happened
        # one must always be selected
        self.failUnless( len(sinc.selectedfeatures) > 0 )
        # mask have origshape
        self.failUnless( sinc.selectionmask.shape ==\
                         mask.shape )

        print sinc.cvperf
        print sinc.cvperf.mean()
        print stats.ttest_1samp(sinc.cvperf, 0.5)


def suite():
    return unittest.makeSuite(SerialInclusionTests)


if __name__ == '__main__':
    unittest.main()

