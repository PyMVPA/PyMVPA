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
from sets import Set
from mvpa.datasets.dataset import Dataset
from mvpa.datasets.misc import zscore, aggregateFeatures
from mvpa.datasets.maskmapper import MaskMapper
from mvpa.misc.exceptions import DatasetError

class DatasetTests(unittest.TestCase):

    def testAddPatterns(self):
        data = Dataset(samples=range(5), labels=1, chunks=1)

        self.failUnlessEqual(
            data.uniquelabels, [1],
            msg="uniquelabels must be correctly recomputed")

        # simple sequence has to be a single pattern
        self.failUnlessEqual( data.nsamples, 1)
        # check correct pattern layout (1x5)
        self.failUnless(
            (data.samples == N.array([[0, 1, 2, 3, 4]])).all() )

        # check for single labels and origin
        self.failUnless( (data.labels == N.array([1])).all() )
        self.failUnless( (data.chunks == N.array([1])).all() )

        # now try adding pattern with wrong shape
        self.failUnlessRaises( DatasetError,
                               data.__iadd__, Dataset(samples=N.ones((2,3)),
                                                      labels=1,
                                                      chunks=1))

        # now add two real patterns
        data += Dataset(samples=N.random.standard_normal((2,5)),
                        labels=2, chunks=2 )
        self.failUnlessEqual( data.nfeatures, 5 )
        self.failUnless( (data.labels == N.array([1,2,2]) ).all() )
        self.failUnless( (data.chunks == N.array([1,2,2]) ).all() )

        # test automatic origins
        data += Dataset(samples=N.random.standard_normal((2,5)), labels=3)
        self.failUnless( (data.chunks == N.array([1,2,2,0,1]) ).all() )

        # test unique class labels
        self.failUnless( (data.uniquelabels == N.array([1,2,3]) ).all() )

        # test wrong label length
        self.failUnlessRaises( DatasetError,
                               Dataset,
                               samples=N.random.standard_normal((4,5)),
                               labels=[ 1, 2, 3 ],
                               chunks=2 )

        # test wrong origin length
        self.failUnlessRaises( DatasetError,
                               Dataset,
                               samples=N.random.standard_normal((4,5)),
                               labels=[ 1, 2, 3, 4 ],
                               chunks=[ 2, 2, 2 ] )


    def testFeatureSelection(self):
        origdata = N.random.standard_normal((10,100))
        data = Dataset(samples=origdata, labels=2, chunks=2 )

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


    def testSampleSelection(self):
        origdata = N.random.standard_normal((10,100))
        data = Dataset(samples=origdata, labels=2, chunks=2 )

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

        # check selection by labels
        sel = data.idsbylabels(2)
        self.failUnless( len(sel) == data.nsamples )

        # not present label
        sel = data.idsbylabels(3)
        self.failUnless( len(sel) == 0 )

        data = Dataset(samples=origdata, labels=[8, 9, 4, 3, 3, 3, 4, 2, 8, 9],
                       chunks=2)
        self.failUnless( (data.idsbylabels([2, 3]) == \
                          [ 3.,  4.,  5.,  7.]).all() )

        # lets cause it to compute unique labels
        self.failUnless( (data.uniquelabels == [2, 3, 4, 8, 9]).all() );


        # select some samples removing some labels completely
        sel = data.selectSamples(data.idsbylabels([3, 4, 8, 9]))
        self.failUnlessEqual(Set(sel.uniquelabels), Set([3, 4, 8, 9]))


    def testCombinedPatternAndFeatureMasking(self):
        data = Dataset(samples=N.arange( 20 ).reshape( (4,5) ),
                       labels=1,
                       chunks=1)

        self.failUnless( data.nsamples == 4 )
        self.failUnless( data.nfeatures == 5 )
        fsel = data.selectFeatures([1,2])
        fpsel = fsel.selectSamples([0,3])
        self.failUnless( fpsel.nsamples == 2 )
        self.failUnless( fpsel.nfeatures == 2 )

        self.failUnless( (fpsel.samples == [[1,2],[16,17]]).all() )


    def testPatternMerge(self):
        data1 = Dataset(samples=N.ones((5,5)), labels=1, chunks=1 )
        data2 = Dataset(samples=N.ones((3,5)), labels=2, chunks=1 )

        merged = data1 + data2

        self.failUnless( merged.nfeatures == 5 )
        self.failUnless( (merged.labels == [ 1,1,1,1,1,2,2,2]).all() )
        self.failUnless( (merged.chunks == [ 1,1,1,1,1,1,1,1]).all() )

        data1 += data2

        self.failUnless( data1.nfeatures == 5 )
        self.failUnless( (data1.labels == [ 1,1,1,1,1,2,2,2]).all() )
        self.failUnless( (data1.chunks == [ 1,1,1,1,1,1,1,1]).all() )


    def testLabelRandomizationAndSampling(self):
        data = Dataset(samples=N.ones((5,1)), labels=range(5), chunks=1 )
        data += Dataset(samples=N.ones((5,1))+1, labels=range(5), chunks=2 )
        data += Dataset(samples=N.ones((5,1))+2, labels=range(5), chunks=3 )
        data += Dataset(samples=N.ones((5,1))+3, labels=range(5), chunks=4 )
        data += Dataset(samples=N.ones((5,1))+4, labels=range(5), chunks=5 )
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
        data2 = Dataset(samples=data.samples,
                        labels=data.labels,
                        chunks=data.chunks )

        # labels are the same as the originals
        self.failUnless( (data2.labels == origlabels).all() )

        # now permute in the new object
        data2.permuteLabels( True )

        # must not affect the old one
        self.failUnless( (data.labels == origlabels).all() )
        # but only the new one
        self.failIf( (data2.labels == origlabels).all() )


    def testAttributes(self):
        #class BlobbyDataset(Dataset):
        #    pass
        # TODO: we can't assign attributes to those for now...
        ds = Dataset(samples=range(5), labels=1, chunks=1)
        self.failUnlessRaises(AttributeError, lambda x:x.blobs, ds)
        """Dataset.blobs should fail since .blobs wasn't yet registered"""

        #register new attribute but it would alter only new instances
        Dataset._registerAttribute("blobs", "_data", hasunique=True)
        ds = Dataset(samples=range(5), labels=1, chunks=1)
        self.failUnless(not ds.blobs != [ 0 ],
                        msg="By default new attributes supposed to get 0 as the value")

        try:
            ds.blobs = [1,2]
            self.fail(msg="Dataset.blobs=[1,2] should fail since there is 5 samples")
        except ValueError, e:
            pass

        try:
            ds.blobs = [1]
        except  e:
            self.fail(msg="We must be able to assign the attribute")

        # Dataset still shouldn't have blobs... just BlobbyDataset
        #self.failUnlessRaises(AttributeError, lambda x:x.blobs,
        #                      Dataset(samples=range(5), labels=1, chunks=1))


    def testRequiredAtrributes(self):
        self.failUnlessRaises(DatasetError, Dataset)
        self.failUnlessRaises(DatasetError, Dataset, samples=[1])
        self.failUnlessRaises(DatasetError, Dataset, labels=[1])
        try:
            ds = Dataset(samples=[1], labels=[1])
        except:
            self.fail(msg="samples and labels are 2 required parameters")

    def testZScoring(self):
        # dataset: mean=2, std=1
        samples = N.array( (0,1,3,4,2,2,3,1,1,3,3,1,2,2,2,2) ).\
            reshape((16, 1))
        data = Dataset(samples=samples,
                       labels=range(16), chunks=[0]*16)
        self.failUnlessEqual( data.samples.mean(), 2.0 )
        self.failUnlessEqual( data.samples.std(), 1.0 )
        zscore(data, perchunk=True)

        # check z-scoring
        check = N.array([-2,-1,1,2,0,0,1,-1,-1,1,1,-1,0,0,0,0],
                        dtype='float64').reshape(16,1)
        self.failUnless( (data.samples ==  check).all() )

        data = Dataset(samples=samples,
                       labels=range(16), chunks=[0]*16)
        zscore(data, perchunk=False)
        self.failUnless( (data.samples ==  check).all() )

        # check z-scoring taking set of labels as a baseline
        data = Dataset(samples=samples,
                       labels=[0, 2, 2, 2, 1] + [2]*11,
                       chunks=[0]*16)
        zscore(data, baselinelabels=[0, 1])
        self.failUnless((samples == data.samples+1.0).all())


    def testAggregation(self):
        data = Dataset(samples=N.arange( 20 ).reshape( (4,5) ),
                       labels=1,
                       chunks=1)

        ag_data = aggregateFeatures(data, N.mean)

        self.failUnless(ag_data.nsamples == 4)
        self.failUnless(ag_data.nfeatures == 1)
        self.failUnless((ag_data.samples[:,0] == [2, 7, 12, 17]).all())

    def testApplyMapper(self):
        """Test creation of new dataset by applying a mapper"""
        mapper = MaskMapper(N.array([1,0,1]))
        dataset = Dataset(samples=N.arange(12).reshape( (4,3) ),
                          labels=1,
                          chunks=1)
        seldataset = dataset.applyMapper(featuresmapper=mapper)
        self.failUnless( (dataset.selectFeatures([0, 2]).samples
                          == seldataset.samples).all() )
        self.failUnlessRaises(NotImplementedError,
                              dataset.applyMapper, None, [1])
        """We don't yet have implementation for samplesmapper --
        if we get one -- remove this check and place a test"""

    def testId(self):
        """Test Dataset._id() if it gets changed if any of the labels/chunks changes"""

        dataset = Dataset(samples=N.arange(12).reshape( (4,3) ),
                          labels=1,
                          chunks=1)
        origid = dataset._id
        dataset.labels = [3, 1, 2, 3]           # change all labels
        self.failUnless(origid != dataset._id,
                        msg="Changing all labels should alter dataset's _id")

        origid = dataset._id

        z = dataset.labels[1]
        self.failUnlessEqual(origid, dataset._id,
                             msg="Accessing shouldn't change _id")
        z = dataset.chunks
        self.failUnlessEqual(origid, dataset._id,
                             msg="Accessing shouldn't change _id")
        z[2] = 333
        self.failUnless(origid != dataset._id,
                        msg="Changing value in attribute should change _id")

        origid = dataset._id
        dataset.samples[1,1] = 1000
        self.failUnless(origid != dataset._id,
                        msg="Changing value in data should change _id")


        origid = dataset._id
        dataset.permuteLabels(True)
        self.failUnless(origid != dataset._id,
                        msg="Permutation also changes _id")

        dataset.permuteLabels(False)
        self.failUnless(origid != dataset._id,
                        msg="Permutation also changes _id even on restore")


    def testFeatureMaskConversion(self):
        dataset = Dataset(samples=N.arange(12).reshape( (4,3) ),
                          labels=1,
                          chunks=1)

        mask = dataset.convertFeatureIds2FeatureMask(range(dataset.nfeatures))
        self.failUnless(len(mask) == dataset.nfeatures)
        self.failUnless((mask == True).all())

        self.failUnless(
            (dataset.convertFeatureMask2FeatureIds(mask) == range(3)).all())

        mask[1] = False

        self.failUnless(
            (dataset.convertFeatureMask2FeatureIds(mask) == [0, 2]).all())



def suite():
    return unittest.makeSuite(DatasetTests)


if __name__ == '__main__':
    import test_runner

