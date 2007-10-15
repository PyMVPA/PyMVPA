### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA recursive feature elimination
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

import unittest
import numpy as np
import mvpa.rfe as rfe
import mvpa.maskeddataset
import mvpa.svm

def dumbFeatureSignal():
    data = [[0,1],[1,1],[0,2],[1,2],[0,3],[1,3],[0,4],[1,4],
            [0,5],[1,5],[0,6],[1,6],[0,7],[1,7],[0,8],[1,8],
            [0,9],[1,9],[0,10],[1,10],[0,11],[1,11],[0,12],[1,12]]
    regs = [1 for i in range(8)] \
         + [2 for i in range(8)] \
         + [3 for i in range(8)]

    return mvpa.maskeddataset.MaskedDataset(data, regs, None)


class RFETests(unittest.TestCase):

    def setUp(self):
        self.dumbpattern = dumbFeatureSignal()

        # prepare second demo dataset and mask
        self.mask = np.ones((20))
        data = np.repeat(np.arange(100),20).reshape((100,20))
        # add noise; first remains pristine
        for d in range(data.shape[1]):
            data[:,d] += np.random.normal(0, d + 10, 100)
        reg = np.logical_and(np.arange(100) > 24,
                             np.arange(100) < 75).astype('int')
        reg = (np.arange(100) > 49).astype('int')
        orig = np.arange(100) % 5
        self.pattern = mvpa.maskeddataset.MaskedDataset(data, reg, orig)

    def testFeatureRanking(self):
        obj = rfe.RFE( self.dumbpattern )

        self.failUnless( obj.pattern.nfeatures == 2 )

        # kill the dumb feature
        obj.killNFeatures(1)

        # check that the important is still in
        self.failUnless( obj.pattern.nfeatures == 1)
        self.failUnless( ( obj.pattern.samples[:,0] ==\
                           self.dumbpattern.samples[:,1] ).all() )


    def testSelectFeatures(self):
        obj = rfe.RFE( self.pattern )
        self.failUnless( obj.pattern.nfeatures == 20 )

        obj.selectFeatures(10)
        self.failUnless( obj.pattern.nfeatures == 10 )

        # ensure unchanges original array
        self.failUnless(self.pattern.nfeatures == 20)


    def testKillNFeatures(self):
        obj = rfe.RFE( self.pattern )
        self.failUnless( obj.pattern.nfeatures == 20 )

        obj.killNFeatures(3)
        self.failUnless( obj.pattern.nfeatures == 17 )

        # ensure unchanges original array
        self.failUnless(self.pattern.nfeatures == 20)


    def testKillFeatureFraction(self):
        obj = rfe.RFE( self.pattern )
        self.failUnless( obj.pattern.nfeatures == 20 )

        obj.killFeatureFraction(0.75)
        self.failUnless( obj.pattern.nfeatures == 5 )

        # ensure unchanges original array
        self.failUnless(self.pattern.nfeatures == 20)


    def testDataTesting(self):
        obj = rfe.RFE( self.dumbpattern )
        # kill the dumb feature
        obj.killNFeatures(1)

        # check performance on the training data
        # includes an implicite feature selection with the RFE internal mask
        pred, perf, confmat = obj.testSelection(self.dumbpattern)

        # prediction has to be perfect
        self.failUnless( perf == 1.0 )
        self.failUnless( ( pred == self.dumbpattern.regs ).all() )
        self.failUnless( confmat.shape == (3,3) )


        # make slightly different dataset, but with the same underlying
        # concept
        tdata = \
            mvpa.maskeddataset.MaskedDataset( [[1.5],[2.5],[3.5],
                               [5.5],[6.5],[7.5],
                               [9.5],[10.5],[11.5]],
                               [1 for i in range(3)] \
                               + [2 for i in range(3)] \
                               + [3 for i in range(3)],
                               None )

        # check performance on the new dataset
        # includes a check whether a dataset not matching the original
        # shape can be used, as long as the number of features match
        pred, perf, confmat = obj.testSelection(tdata)
        # prediction has to be perfect
        self.failUnless( perf == 1.0 )
        self.failUnless( ( pred == tdata.regs ).all() )
        self.failUnless( confmat.shape == (3,3) )


    def testEliminationMask(self):
        obj = rfe.RFE( self.dumbpattern )
        # kill the dumb feature
        elim_mask = obj.selectFeatures( 1, 
                                        eliminate_by='number',
                                        kill_per_iter = 1)

        self.failUnless( elim_mask.shape == self.dumbpattern.mapper.dsshape )
        self.failUnless( elim_mask[0] == 0 and elim_mask[1] == 1 )

#        pat = mvpa.MVPAPattern( np.random.normal(0,size=(20,16,16,8)),
#                                [i%2 for i in range(20)],
#                                0 )
#
#        mask = np.ones((16,16,8),dtype='int16')
#        mask[0:4,   0:4,   0:2]=False
#        mask[12:16, 0:4,   0:2]=False
#        mask[0:4,   12:16, 0:2]=False
#        mask[12:16, 12:16, 0:2]=False
#
#        pat_sel = pat.selectFeatures(mask)


    def testEliminationEvenMore(self):

        def absolute_coord2id( coord, dims ):
            """ Calculates the feature id from a coordinate value within certain
            dimensions.
            """
            # transform shape and coordinate into array for easy handling
            ac = np.array( coord )
            ao = np.array( dims )

#            # check for sane coordinates
#            if (ac >= ao).all() \
#               or (ac < numpy.repeat( 0, len( ac ) ) ).all():
#                raise ValueError, 'Invalid coordinate: outside array ' \
#                                  '( coord: %s, arrayshape: %s )' % \
#                                  ( str(coord), str(dims) )

            # compute offsets on each axis
            offsets = [ ac[d] * ao[d+1:].prod() for d in range(len(ao)) ]

            return sum(offsets)

        pat = mvpa.maskeddataset.MaskedDataset( np.random.normal(size=(100,2,3,4)),
                                [i%2 for i in range(100)],
                                [i/5 for i in range(100)] )

        # make a copy of the original patterns
        sec_pat = pat.samples.copy()

        el = rfe.RFE( pat )
        el.killNFeatures(5, eliminate_by = 'number', kill_per_iter = 1 )

        features = np.transpose(el.pattern.mapper.getMask().nonzero())

        for i,f in enumerate(features):
            orig_id = absolute_coord2id(f,pat.mapper.dsshape)
            orig_f = sec_pat[:,orig_id]
            self.failUnless( (el.pattern.samples[:,i] == orig_f).all() )


def suite():
    return unittest.makeSuite(RFETests)


if __name__ == '__main__':
    unittest.main()

