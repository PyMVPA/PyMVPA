#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA pattern handling"""

from mvpa.datasets.maskeddataset import *
from mvpa.misc.exceptions import DatasetError

import unittest
import numpy as N
import random

class MaskedDatasetTests(unittest.TestCase):

    def testCreateMaskedDataset(self):
        data = MaskedDataset(samples=[range(5)], labels=1,
                             chunks=1)
        # simple sequence has to be a single pattern
        self.failUnlessEqual( data.nsamples, 1)
        # check correct pattern layout (1x5)
        self.failUnless(
            (data.samples == N.array([[0, 1, 2, 3, 4]])).all() )

        # check for single label and origin
        self.failUnless( (data.labels == N.array([1])).all() )
        self.failUnless( (data.chunks == N.array([1])).all() )

        # now try adding pattern with wrong shape
        self.failUnlessRaises(DatasetError,
                              data.__iadd__,
                              MaskedDataset(samples=N.ones((2,3)), labels=1,
                                            chunks=1))

        # now add two real patterns
        data += MaskedDataset(samples=N.random.standard_normal((2,5)),
                              labels=2, chunks=2)
        self.failUnlessEqual( data.nsamples, 3 )
        self.failUnless( (data.labels == N.array([1,2,2]) ).all() )
        self.failUnless( (data.chunks == N.array([1,2,2]) ).all() )


        # test unique class labels
        data += MaskedDataset(samples=N.random.standard_normal((2,5)),
                              labels=3)
        self.failUnless( (data.uniquelabels == N.array([1,2,3]) ).all() )

        # test wrong label length
        self.failUnlessRaises(DatasetError,
                              MaskedDataset,
                              samples=N.random.standard_normal((4,2,3,4)),
                              labels=[1, 2, 3],
                              chunks=2)

        # test wrong origin length
        self.failUnlessRaises( DatasetError,
                               MaskedDataset,
                               samples=N.random.standard_normal((4,2,3,4)),
                               labels=[1, 2, 3, 4],
                               chunks=[2, 2, 2])


    def testShapeConversion(self):
        data = MaskedDataset(samples=N.arange(24).reshape((2,3,4)),
                             labels=1, chunks=1)
        self.failUnlessEqual(data.nsamples, 2)
        self.failUnless(data.mapper.dsshape == (3,4))
        self.failUnlessEqual(data.samples.shape, (2,12))
        self.failUnless((data.samples ==
                         N.array([range(12),range(12,24)])).all())


    def testPatternShape(self):
        data = MaskedDataset(samples=N.ones((10,2,3,4)), labels=1, chunks=1)
        self.failUnless(data.samples.shape == (10,24))
        self.failUnless(data.mapper.dsshape == (2,3,4))


    def testFeature2Coord(self):
        origdata = N.random.standard_normal((10,2,4,3,5))
        data = MaskedDataset( samples=origdata, labels=2, chunks=2 )

        def randomCoord(shape):
            return [ random.sample(range(size),1)[0] for size in shape ]

        # check 100 random coord2feature transformations
        for i in xrange(100):
            # choose random coord
            c = randomCoord(data.mapper.dsshape)
            # tranform to feature_id
            id = data.mapper.getOutId(c)

            # compare data from orig array (selected by coord)
            # and data from pattern array (selected by feature id)
            orig = origdata[:,c[0],c[1],c[2],c[3]]
            pat = data.samples[:, id]

            self.failUnless((orig == pat).all())


    def testCoord2Feature(self):
        origdata = N.random.standard_normal((10,2,4,3,5))
        data = MaskedDataset(samples=origdata, labels=2, chunks=2)

        def randomCoord(shape):
            return [ random.sample(range(size),1)[0] for size in shape ]

        for id in xrange(data.nfeatures):
            # transform to coordinate
            c = data.mapper.getInId(id)
            self.failUnlessEqual(len(c), 4)

            # compare data from orig array (selected by coord)
            # and data from pattern array (selected by feature id)
            orig = origdata[:,c[0],c[1],c[2],c[3]]
            pat = data.samples[:, id]

            self.failUnless((orig == pat).all())


    def testFeatureSelection(self):
        origdata = N.random.standard_normal((10,2,4,3,5))
        data = MaskedDataset(samples=origdata, labels=2, chunks=2)

        unmasked = data.samples.copy()

        # default must be no mask
        self.failUnless( data.nfeatures == 120 )
        self.failUnless(data.mapper.getOutSize() == 120)

        # check that full mask uses all features
        sel = data.selectFeaturesByMask( N.ones((2,4,3,5)) )
        self.failUnless( sel.nfeatures == data.samples.shape[1] )
        self.failUnless( data.nfeatures == 120 )

        # check partial array mask
        partial_mask = N.zeros((2,4,3,5), dtype='uint')
        partial_mask[0,0,2,2] = 1
        partial_mask[1,2,2,0] = 1

        sel = data.selectFeaturesByMask( partial_mask )
        self.failUnless( sel.nfeatures == 2 )
        self.failUnless( sel.mapper.getMask().shape == (2,4,3,5))

        # check that feature selection does not change source data
        self.failUnless(data.nfeatures == 120)
        self.failUnlessEqual(data.mapper.getOutSize(), 120)

        # check selection with feature list
        sel = data.selectFeatures([0,37,119])
        self.failUnless(sel.nfeatures == 3)

        # check size of the masked patterns
        self.failUnless( sel.samples.shape == (10,3) )

        # check that the right features are selected
        self.failUnless( (unmasked[:,[0,37,119]]==sel.samples).all() )


    def testPatternSelection(self):
        origdata = N.random.standard_normal((10,2,4,3,5))
        data = MaskedDataset(samples=origdata, labels=2, chunks=2)

        self.failUnless( data.nsamples == 10 )

        # set single pattern to enabled
        sel=data.selectSamples(5)
        self.failUnless( sel.nsamples == 1 )
        self.failUnless( data.nsamples == 10 )

        # check duplicate selections
        sel = data.selectSamples([5,5])
        self.failUnless( sel.nsamples == 2 )
        self.failUnless( (sel.samples[0] == sel.samples[1]).all() )
        self.failUnless( len(sel.labels) == 2 )
        self.failUnless( len(sel.chunks) == 2 )

        self.failUnless( sel.samples.shape == (2,120) )

    def testCombinedPatternAndFeatureMasking(self):
        data = MaskedDataset(
            samples=N.arange( 20 ).reshape( (4,5) ), labels=1, chunks=1 )

        self.failUnless( data.nsamples == 4 )
        self.failUnless( data.nfeatures == 5 )
        fsel = data.selectFeatures([1,2])
        fpsel = fsel.selectSamples([0,3])
        self.failUnless( fpsel.nsamples == 2 )
        self.failUnless( fpsel.nfeatures == 2 )

        self.failUnless( (fpsel.samples == [[1,2],[16,17]]).all() )


    def testOrigMaskExtraction(self):
        origdata = N.random.standard_normal((10,2,4,3))
        data = MaskedDataset(samples=origdata, labels=2, chunks=2)

        # check with custom mask
        sel = data.selectFeatures([5])
        self.failUnless( sel.samples.shape[1] == 1 )
        origmask = sel.mapper.getMask()
        self.failUnless( origmask[0,1,2] == True )
        self.failUnless( origmask.shape == data.mapper.dsshape == (2,4,3) )



    def testPatternMerge(self):
        data1 = MaskedDataset(samples=N.ones((5,5,1)), labels=1, chunks=1)
        data2 = MaskedDataset(samples=N.ones((3,5,1)), labels=2, chunks=1)

        merged = data1 + data2

        self.failUnless(merged.nsamples == 8 )
        self.failUnless((merged.labels == [ 1,1,1,1,1,2,2,2]).all())
        self.failUnless((merged.chunks == [ 1,1,1,1,1,1,1,1]).all())

        data1 += data2

        self.failUnless(data1.nsamples == 8 )
        self.failUnless((data1.labels == [ 1,1,1,1,1,2,2,2]).all())
        self.failUnless((data1.chunks == [ 1,1,1,1,1,1,1,1]).all())


    def testLabelRandomizationAndSampling(self):
        data = MaskedDataset(samples=N.ones((5,1)), labels=range(5), chunks=1)
        data += MaskedDataset(samples=N.ones((5,1))+1, labels=range(5), chunks=2)
        data += MaskedDataset(samples=N.ones((5,1))+2, labels=range(5), chunks=3)
        data += MaskedDataset(samples=N.ones((5,1))+3, labels=range(5), chunks=4)
        data += MaskedDataset(samples=N.ones((5,1))+4, labels=range(5), chunks=5)
        self.failUnless( data.samplesperlabel == {0:5, 1:5, 2:5, 3:5, 4:5} )

        sample = data.getRandomSamples( 2 )
        self.failUnless( sample.samplesperlabel.values() == [ 2,2,2,2,2 ] )

        self.failUnless( (data.uniquechunks == range(1,6)).all() )

        # store the old labels
        origlabels = data.labels.copy()

        data.permuteLabels(True)

        self.failIf( (data.labels == origlabels).all() )

        data.permuteLabels(False)

        self.failUnless( (data.labels == origlabels).all() )

        # now try another object with the same data
        data2 = MaskedDataset(samples=data.samples, labels=data.labels,
                              chunks=data.chunks )

        # labels are the same as the originals
        self.failUnless( (data2.labels == origlabels).all() )

        # now permute in the new object
        data2.permuteLabels( True )

        # must not affect the old one
        self.failUnless( (data.labels == origlabels).all() )
        # but only the new one
        self.failIf( (data2.labels == origlabels).all() )


    def testFeatureMasking(self):
        mask = N.zeros((5,3),dtype='bool')
        mask[2,1] = True; mask[4,0] = True
        data = MaskedDataset(
            samples=N.arange( 60 ).reshape( (4,5,3) ), labels=1, chunks=1,
            mask=mask)

        # check simple masking
        self.failUnless( data.nfeatures == 2 )
        self.failUnless( data.mapper.getOutId( (2,1) ) == 0 
                     and data.mapper.getOutId( (4,0) ) == 1 )
        self.failUnlessRaises( ValueError, data.mapper.getOutId, (2,3) )
        self.failUnless( data.mapper.getMask().shape == (5,3) )
        self.failUnless( tuple(data.mapper.getInId( 1 )) == (4,0) )

        # selection should be idempotent
        self.failUnless(data.selectFeaturesByMask( mask ).nfeatures == data.nfeatures )
        # check that correct feature get selected
        self.failUnless( (data.selectFeatures([1]).samples[:,0] \
                          == N.array([12, 27, 42, 57]) ).all() )
        self.failUnless(tuple( data.selectFeatures([1]).mapper.getInId(0) ) == (4,0) )
        self.failUnless( data.selectFeatures([1]).mapper.getMask().sum() == 1 )


#    def testROIMasking(self):
#        mask=N.array([i/6 for i in range(60)], dtype='int').reshape(6,10)
#        data = MaskedDataset(
#            N.arange( 180 ).reshape( (3,6,10) ), 1, 1, mask=mask )
#
#        self.failIf( data.mapper.getMask().dtype == 'bool' )
#        # check that the 0 masked features get cut
#        self.failUnless( data.nfeatures == 54 )
#        self.failUnless( (data.samples[:,0] == [6,66,126]).all() )
#        self.failUnless( data.mapper.getMask().shape == (6,10) )
#
#        featsel = data.selectFeatures([19])
#        self.failUnless( (data.samples[:,19] == featsel.samples[:,0]).all() )
#
#        # check single ROI selection works
#        roisel = data.selectFeaturesByGroup([4])
#        self.failUnless( (data.samples[:,19] == roisel.samples[:,1]).all() )
#
#        # check dual ROI selection works (plus keep feature order)
#        roisel = data.selectFeaturesByGroup([6,4])
#        self.failUnless( (data.samples[:,19] == roisel.samples[:,1]).all() )
#        self.failUnless( (data.samples[:,32] == roisel.samples[:,8]).all() )
#
#        # check if feature coords can be recovered
#        self.failUnless( (roisel.getCoordinate(8) == (3,8)).all() )


def suite():
    return unittest.makeSuite(MaskedDatasetTests)


if __name__ == '__main__':
    import test_runner

