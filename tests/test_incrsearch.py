### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA incremental feature search algorithm
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

import mvpa.incrsearch as si
import mvpa
import mvpa.knn as knn
import mvpa.svm_wrap as svm_wrap
import unittest
import numpy as np
import scipy.stats as stats

class IncrementalSearchTests(unittest.TestCase):

    def setUp(self):
        # prepare demo dataset and mask
        self.mask = np.ones((3,3,3))
        data = np.random.uniform(0,1,(100,) + (self.mask.shape))
        reg = np.arange(100) / 50
        orig = range(10) * 10
        self.pattern = mvpa.MVPAPattern(data, reg, orig)



    def testIncrementalSearch(self):
        # init algorithm
        sinc = si.IncrementalFeatureSearch( self.pattern,
                                          self.mask,
                                          ncvfoldsamples=1 )

        # run selection of single features
        selected_features = sinc.selectFeatures( knn.kNN, verbose=True)

        # no real check yet, simply checking something happened
        # one must always be selected
        self.failUnless( len(selected_features) > 0 )
        # mask have origshape
        self.failUnless( sinc.selectionmask.shape == \
                         self.mask.shape )


    def testIncrementalROISearch(self):
        # make 3 slice roi mask
        self.mask[0] = 1
        self.mask[1] = 2
        self.mask[2] = 3

        # init algorithm
        sinc = si.IncrementalFeatureSearch( self.pattern,
                                          self.mask,
                                          ncvfoldsamples=1 )

        # run selection of single features
        selected_rois = sinc.selectROIs( knn.kNN, verbose=True)

        # only three ROIs max
        self.failUnless( len( selected_rois ) <= 3 )
        # each ROI has 9 so total selection number is dividable by 9
        self.failUnless( np.sum(sinc.selectionmask) % 9 == 0 )


def suite():
    return unittest.makeSuite(IncrementalSearchTests)


if __name__ == '__main__':
    unittest.main()

