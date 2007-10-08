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

import mvpa.ifs as si
import mvpa
import mvpa.svm as svm
import unittest
import numpy as N

class IncrementalSearchTests(unittest.TestCase):

    def setUp(self):
        # prepare demo dataset and mask
        self.mask = N.ones((3,3,3))
        data = N.random.uniform(0,1,(100,) + (self.mask.shape))
        reg = N.arange(100) / 50
        orig = range(10) * 10
        self.pattern = mvpa.MVPAPattern(data, reg, orig)



    def testIncrementalSearch(self):
        # init algorithm
        sinc = si.IFS()

        # run selection of single features
        selected_features, rank_map = sinc.selectFeatures( self.pattern,
                                                 svm.SVM(),
                                                 ncvfoldsamples=1)

        # no real check yet, simply checking something happened
        # one must always be selected
        self.failUnless( selected_features.nfeatures >= 1 )

        self.failUnless( rank_map.shape == self.pattern.origshape )
        self.failIf( (rank_map == 0.0).any() )

        # mask have origshape
        self.failUnless( selected_features.origshape == \
                         self.mask.shape )

        sinc.ntoselect = 5
        selected_features, rank_map = \
            sinc.selectFeatures( self.pattern,
                                 svm.SVM(),
                                 ncvfoldsamples=1)

        # five must always be selected
        self.failUnless( selected_features.nfeatures >= 5 )

#    def testIncrementalROISearch(self):
#        # make 3 slice roi mask
#        self.mask[0] = 1
#        self.mask[1] = 2
#        self.mask[2] = 3
#
#        # init algorithm
#        sinc = si.IncrementalFeatureSearch( self.pattern,
#                                          self.mask,
#                                          ncvfoldsamples=1 )
#
#        # run selection of single features
#        selected_rois = sinc.selectROIs( knn.kNN, verbose=False)
#
#        # no real check yet, simply checking something happened
#        # one must always be selected
#        self.failUnless( len(selected_rois) == 10 )
#        for r in selected_rois:
#            self.failUnless( len(r) >= 1 )
#            # only three ROIs max
#            self.failUnless( len(r) <= 3 )
#
#        res = sinc.getMeanSelectionMask()
#
#        # mask have origshape
#        self.failUnless( res.shape == self.mask.shape )
#
#        for m in sinc.selectionmasks:
#            # each ROI has 9 so total selection number is dividable by 9
#            self.failUnless( N.sum(m) % 9 == 0 )


def suite():
    return unittest.makeSuite(IncrementalSearchTests)


if __name__ == '__main__':
    unittest.main()

