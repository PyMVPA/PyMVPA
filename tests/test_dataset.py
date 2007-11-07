#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dataset handling"""

import unittest
import random
import numpy as N
from mvpa.datasets.dataset import Dataset
from mvpa.datasets.misc import zscore

class DatasetTests(unittest.TestCase):

    def testAddPatterns(self):
        data = Dataset(range(5), 1, 1)
        # simple sequence has to be a single pattern
        self.failUnlessEqual( data.nsamples, 1)
        # check correct pattern layout (1x5)
        self.failUnless(
            (data.samples == N.array([[0, 1, 2, 3, 4]])).all() )

        # check for single labels and origin
        self.failUnless( (data.labels == N.array([1])).all() )
        self.failUnless( (data.chunks == N.array([1])).all() )

        # now try adding pattern with wrong shape
        self.failUnlessRaises( ValueError,
                               data.__iadd__, Dataset(N.ones((2,3)), 1, 1))

        # now add two real patterns
        data += Dataset(N.random.standard_normal((2,5)), 2, 2 )
        self.failUnlessEqual( data.nfeatures, 5 )
        self.failUnless( (data.labels == N.array([1,2,2]) ).all() )
        self.failUnless( (data.chunks == N.array([1,2,2]) ).all() )


        # test automatic origins
        data += Dataset( N.random.standard_normal((2,5)), 3, None )
        self.failUnless( (data.chunks == N.array([1,2,2,0,1]) ).all() )

        # test unique class labels
        self.failUnless( (data.uniquelabels == N.array([1,2,3]) ).all() )

        # test wrong label length
        self.failUnlessRaises( ValueError,
                               Dataset,
                               N.random.standard_normal((4,5)),
                               [ 1, 2, 3 ],
                               2 )

        # test wrong origin length
        self.failUnlessRaises( ValueError,
                               Dataset,
                               N.random.standard_normal((4,5)),
                               [ 1, 2, 3, 4 ],
                               [ 2, 2, 2 ] )


    def testFeatureSelection(self):
        origdata = N.random.standard_normal((10,100))
        data = Dataset( origdata, 2, 2 )

        unmasked = data.samples.copy()

        # default must be no mask
        self.failUnless( data.nfeatures == 100 )

        # check selection with feature list
        sel = data.selectFeatures( [0,20,79] )
        self.failUnless(sel.nfeatures == 3)

        # check size of the masked patterns
        self.failUnless( sel.samples.shape == (10,3) )

        # check that the right features are selected
        self.failUnless( (unmasked[:,[0,20,79]]==sel.samples).all() )


    def testPatternSelection(self):
        origdata = N.random.standard_normal((10,100))
        data = Dataset( origdata, 2, 2 )

        self.failUnless( data.nsamples == 10 )

        # set single pattern to enabled
        sel=data.selectSamples(5)
        self.failUnless( sel.nsamples == 1 )
        self.failUnless( data.nfeatures == 100 )

        # check duplicate selections
        sel = data.selectSamples([5,5])
        self.failUnless( sel.nsamples == 2 )
        self.failUnless( (sel.samples[0] == sel.samples[1]).all() )
        self.failUnless( len(sel.labels) == 2 )
        self.failUnless( len(sel.chunks) == 2 )

        self.failUnless( sel.samples.shape == (2,100) )

        # # check selection by labels
        # sel = data.selectSamplesByLabels(2)
        # self.failUnless( sel.nsamples == data.nsamples )
		#
        # # not present label
        # sel = data.selectSamplesByLabels(3)
        # self.failUnless( sel.nsamples == 0 )

        data = Dataset(origdata, [8, 9, 4, 3, 3, 3, 4, 2, 8, 9], 2)
        self.failUnless( (data.getSampleIdsByLabels([2, 3]) == \
                          [ 3.,  4.,  5.,  7.]).all() )


    def testCombinedPatternAndFeatureMasking(self):
        data = Dataset(N.arange( 20 ).reshape( (4,5) ), 1, 1)

        self.failUnless( data.nsamples == 4 )
        self.failUnless( data.nfeatures == 5 )
        fsel = data.selectFeatures([1,2])
        fpsel = fsel.selectSamples([0,3])
        self.failUnless( fpsel.nsamples == 2 )
        self.failUnless( fpsel.nfeatures == 2 )

        self.failUnless( (fpsel.samples == [[1,2],[16,17]]).all() )


    def testPatternMerge(self):
        data1 = Dataset( N.ones((5,5)), 1, 1 )
        data2 = Dataset( N.ones((3,5)), 2, 1 )

        merged = data1 + data2

        self.failUnless( merged.nfeatures == 5 )
        self.failUnless( (merged.labels == [ 1,1,1,1,1,2,2,2]).all() )
        self.failUnless( (merged.chunks == [ 1,1,1,1,1,1,1,1]).all() )

        data1 += data2

        self.failUnless( data1.nfeatures == 5 )
        self.failUnless( (data1.labels == [ 1,1,1,1,1,2,2,2]).all() )
        self.failUnless( (data1.chunks == [ 1,1,1,1,1,1,1,1]).all() )


    def testRegressorRandomizationAndSampling(self):
        data = Dataset( N.ones((5,1)), range(5), 1 )
        data += Dataset( N.ones((5,1))+1, range(5), 2 )
        data += Dataset( N.ones((5,1))+2, range(5), 3 )
        data += Dataset( N.ones((5,1))+3, range(5), 4 )
        data += Dataset( N.ones((5,1))+4, range(5), 5 )

        self.failUnless( data.samplesperlabel == [ 5,5,5,5,5 ] )

        sample = data.getRandomSamples( 2 )

        self.failUnless( sample.samplesperlabel == [ 2,2,2,2,2 ] )

        self.failUnless( (data.uniquechunks == range(1,6)).all() )

        # store the old labels
        origlabels = data.labels.copy()

        data.permutedRegressors(True)

        self.failIf( (data.labels == origlabels).all() )

        data.permutedRegressors(False)

        self.failUnless( (data.labels == origlabels).all() )

        # now try another object with the same data
        data2 = Dataset( data.samples, data.labels, data.chunks )

        # labels are the same as the originals
        self.failUnless( (data2.labels == origlabels).all() )

        # now permute in the new object
        data2.permutedRegressors( True )

        # must not affect the old one
        self.failUnless( (data.labels == origlabels).all() )
        # but only the new one
        self.failIf( (data2.labels == origlabels).all() )


    def testZScoring(self):
        # dataset: mean=2, std=1
        samples = N.array( (0,1,3,4,2,2,3,1,1,3,3,1,2,2,2,2) ).\
            reshape((16, 1))
        data = Dataset(samples,
                       range(16), [0]*16)
        self.failUnlessEqual( data.samples.mean(), 2.0 )
        self.failUnlessEqual( data.samples.std(), 1.0 )
        zscore(data, perchunk=True)

        # check z-scoring
        check = N.array([-2,-1,1,2,0,0,1,-1,-1,1,1,-1,0,0,0,0],
                        dtype='float64').reshape(16,1)
        self.failUnless( (data.samples ==  check).all() )

        data = Dataset(samples,
                       range(16), [0]*16)
        zscore(data, perchunk=False)
        self.failUnless( (data.samples ==  check).all() )

        # check z-scoring taking set of labels as a baseline
        data = Dataset(samples,
                       [0, 2, 2, 2, 1] + [2]*11,
                       [0]*16)
        zscore(data, baselinelabels=[0, 1])
        self.failUnless((samples == data.samples+1.0).all())


def suite():
    return unittest.makeSuite(DatasetTests)


if __name__ == '__main__':
    import test_runner

