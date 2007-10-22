### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA incremental feature search algorithm
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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import mvpa, mvpa.magsel, mvpa.svm, mvpa.featsel
import unittest
import numpy as N

class MagSelTests(unittest.TestCase):

    def setUp(self):
        # prepare demo dataset and mask
        self.mask = N.ones((3,3,3))
        data = N.random.uniform(0,1,(100,) + (self.mask.shape))
        reg = N.arange(100) / 50
        orig = range(10) * 10
        self.pattern = mvpa.MVPAPattern(data, reg, orig)



    def testMagSel(self):
        # init algorithm
        fselect = \
            mvpa.magsel.MagnitudeFeatureSelection( 10,
                                                   select_by = 'number',
                                                   verbose=True )

        # run selection of single features
        selected, frating = fselect.selectFeatures( self.pattern,
                                                    mvpa.svm.SVM() )

        self.failUnless( selected.nfeatures == 10 )

        self.failUnless( frating.shape == self.pattern.origshape )
        self.failIf( (frating == 0.0).any() )

        # mask have origshape
        self.failUnless( selected.origshape == \
                         self.mask.shape )


    def testCVMagSel(self):
        clf = mvpa.svm.SVM( kernel_type=mvpa.svm.libsvm.LINEAR,
                            svm_type = mvpa.svm.libsvm.NU_SVC,
                            nu = 0.7,
                            eps = 0.1 )
        fselect = \
            mvpa.magsel.MagnitudeFeatureSelection( 10,
                                                   select_by = 'number',
                                                   verbose=True )
        fsv = mvpa.featsel.FeatSelValidation( self.pattern, cvtype=1 )
        fsv( fselect, clf )

        print fsv.perfs
        print N.mean(fsv.perfs)
        print fsv.getMeanRatingMap()

        clf.train(self.pattern)
        print clf.getFeatureBenchmark()


def suite():
    return unittest.makeSuite(MagSelTests)


if __name__ == '__main__':
    unittest.main()

