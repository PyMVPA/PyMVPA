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
from mvpa.datasets import Dataset
from mvpa.datasets.miscfx import zscore, aggregateFeatures
from mvpa.mappers.mask import MaskMapper
from mvpa.misc.exceptions import DatasetError

from tests_warehouse import datasets

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

        # define some feature groups
        data.defineFeatureGroups(N.repeat(range(4), 25))

        unmasked = data.samples.copy()

        # default must be no mask
        self.failUnless( data.nfeatures == 100 )

        bsel = N.array([False]*100)
        bsel[ [0,20,79] ] = True
        # check selection with feature list
        for sel in [ data.selectFeatures( [0,20,79], sort=False ),
                     data.select(slice(None), [0,20,79]),
                     data.select(slice(None), N.array([0,20,79])),
                     data.select(slice(None), bsel),
                     ]:
            self.failUnless(sel.nfeatures == 3)

            # check size of the masked patterns
            self.failUnless( sel.samples.shape == (10,3) )

            # check that the right features are selected
            self.failUnless( (unmasked[:,[0,20,79]]==sel.samples).all() )

            # check grouping information
            self.failUnless((sel._dsattr['featuregroups'] == [0, 0, 3]).all())

        # check selection by feature group id
        gsel = data.selectFeatures(groups=[2,3])
        self.failUnless(gsel.nfeatures == 50)


    def testSampleSelection(self):
        origdata = N.random.standard_normal((10,100))
        data = Dataset(samples=origdata, labels=2, chunks=2 )

        self.failUnless( data.nsamples == 10 )

        # set single pattern to enabled
        for sel in [ data.selectSamples(5),
                     data.select(5),
                     data.select(slice(5, 6)),
                     ]:
            self.failUnless( sel.nsamples == 1 )
            self.failUnless( data.nfeatures == 100 )
            self.failUnless( sel.origids == [5] )

        # check duplicate selections
        for sel in [ data.selectSamples([5, 5]),
                     # Following ones would fail since select removes
                     # repetitions (XXX)
                     #data.select([5,5]),
                     #data.select([5,5], 'all'),
                     #data.select([5,5], slice(None)),
                     ]:
            self.failUnless( sel.nsamples == 2 )
            self.failUnless( (sel.samples[0] == data.samples[5]).all() )
            self.failUnless( (sel.samples[0] == sel.samples[1]).all() )
            self.failUnless( len(sel.labels) == 2 )
            self.failUnless( len(sel.chunks) == 2 )
            self.failUnless((sel.origids == [5, 5]).all())

            self.failUnless( sel.samples.shape == (2,100) )

        # check selection by labels
        for sel in [ data.selectSamples(data.idsbylabels(2)),
                     data.select(labels=2),
                     data.select('labels', 2),
                     data.select('labels', [2]),
                     data['labels', [2]]
                     ]:
            self.failUnless( sel.nsamples == data.nsamples )
            self.failUnless( N.all(sel.samples == data.samples) )

        # not present label
        for sel in [ data.selectSamples(data.idsbylabels(3)),
                     data.select(labels=3),
                     data.select('labels', 3),
                     data.select('labels', [3]),
                     ]:
            self.failUnless( sel.nsamples == 0 )

        data = Dataset(samples=origdata,
                       labels=[8, 9, 4, 3, 3, 3, 4, 2, 8, 9],
                       chunks=2)
        for sel in [ data.selectSamples(data.idsbylabels([2, 3])),
                     data.select('labels', [2, 3]),
                     data.select('labels', [2, 3], labels=[1, 2, 3, 4]),
                     data.select('labels', [2, 3], chunks=[1, 2, 3, 4])
                     ]:
            self.failUnless(N.all(sel.origids == [ 3.,  4.,  5.,  7.]))

        # lets cause it to compute unique labels
        self.failUnless( (data.uniquelabels == [2, 3, 4, 8, 9]).all() );


        # select some samples removing some labels completely
        sel = data.selectSamples(data.idsbylabels([3, 4, 8, 9]))
        self.failUnlessEqual(Set(sel.uniquelabels), Set([3, 4, 8, 9]))
        self.failUnless((sel.origids == [0, 1, 2, 3, 4, 5, 6, 8, 9]).all())


    def testEvilSelects(self):
        """Test some obscure selections of samples via select() or __getitem__
        """
        origdata = N.random.standard_normal((10,100))
        data = Dataset(samples=origdata,
                       #       0  1  2  3  4  5  6  7  8  9
                       labels=[8, 9, 4, 3, 3, 3, 3, 2, 8, 9],
                       chunks=[1, 2, 3, 2, 3, 1, 5, 6, 3, 6])

        # malformed getitem
        if __debug__:
            # check is enforced only in __debug__
            self.failUnlessRaises(ValueError, data.__getitem__,
                                  'labels', 'featu')

        # too many indicies
        self.failUnlessRaises(ValueError, data.__getitem__, 1, 1, 1)

        # various getitems which should carry the same result
        for sel in [ data.select('chunks', [2, 6], labels=[3, 2],
                                 features=slice(None)),
                     data.select('all', 'all', labels=[2,3], chunks=[2, 6]),
                     data['chunks', [2, 6], 'labels', [3, 2]],
                     data[:, :, 'chunks', [2, 6], 'labels', [3, 2]],
                     # get warnings but should work as the rest for now
                     data[3:8, 'chunks', [2, 6, 2, 6], 'labels', [3, 2]],
                     ]:
            self.failUnless(N.all(sel.origids == [3,7]))
            self.failUnless(sel.nfeatures == 100)
            self.failUnless(N.all(sel.samples == origdata[ [3,7] ]))

        target = origdata[ [3, 7] ]
        target = target[:, [1,3] ]
        # various getitems which should carry the same result
        for sel in [ data.select('all', [1, 3],
                                 'chunks', [2, 6], labels=[3, 2]),
                     data[:, [1,3], 'chunks', [2, 6], 'labels', [3, 2]],
                     data[:, [1,3], 'chunks', [2, 6], 'labels', [3, 2]],
                     # get warnings but should work as the rest for now
                     data[3:8, [1, 1, 3, 1],
                          'chunks', [2, 6, 2, 6], 'labels', [3, 2]],
                     ]:
            self.failUnless(N.all(sel.origids == [3,7]))
            self.failUnless(sel.nfeatures == 2)
            self.failUnless(N.all(sel.samples == target))

        # Check if we get empty selection if requesting impossible
        self.failUnless(data.select(chunks=[23]).nsamples == 0)

        # Check .where()
        self.failUnless(N.all(data.where(chunks=[2,6]) == [1, 3, 7, 9]))
        self.failUnless(N.all(data.where(chunks=[2,6], labels=[22, 3]) == [3]))
        # both samples and features
        idx = data.where('all', [1, 3, 10], labels=[2, 3, 4])
        self.failUnless(N.all(idx[1] == [1, 3, 10]))
        self.failUnless(N.all(idx[0] == range(2, 8)))
        # empty query
        self.failUnless(data.where() is None)
        # empty result
        self.failUnless(data.where(labels=[123]) == [])


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
        # XXX: the intended test is added as SampleGroupMapper test
#        self.failUnlessRaises(NotImplementedError,
#                              dataset.applyMapper, None, [1])
#        """We don't yet have implementation for samplesmapper --
#        if we get one -- remove this check and place a test"""


    def testId(self):
        """Test Dataset.idhash() if it gets changed if any of the labels/chunks changes"""

        dataset = Dataset(samples=N.arange(12).reshape( (4,3) ),
                          labels=1,
                          chunks=1)
        origid = dataset.idhash
        dataset.labels = [3, 1, 2, 3]           # change all labels
        self.failUnless(origid != dataset.idhash,
                        msg="Changing all labels should alter dataset's idhash")

        origid = dataset.idhash

        z = dataset.labels[1]
        self.failUnlessEqual(origid, dataset.idhash,
                             msg="Accessing shouldn't change idhash")
        z = dataset.chunks
        self.failUnlessEqual(origid, dataset.idhash,
                             msg="Accessing shouldn't change idhash")
        z[2] = 333
        self.failUnless(origid != dataset.idhash,
                        msg="Changing value in attribute should change idhash")

        origid = dataset.idhash
        dataset.samples[1,1] = 1000
        self.failUnless(origid != dataset.idhash,
                        msg="Changing value in data should change idhash")


        origid = dataset.idhash
        dataset.permuteLabels(True)
        self.failUnless(origid != dataset.idhash,
                        msg="Permutation also changes idhash")

        dataset.permuteLabels(False)
        self.failUnless(origid == dataset.idhash,
                        msg="idhash should be restored after permuteLabels(False)")


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


    def testSummary(self):
        """Dummy test"""
        ds = datasets['uni2large']
        ds = ds[N.random.permutation(range(ds.nsamples))[:20]]
        summary = ds.summary()
        self.failUnless(len(summary)>40)

    def testLabelsMapping(self):
        od = {'apple':0, 'orange':1}
        samples = [[3],[2],[3]]
        labels_l = ['apple', 'orange', 'apple']

        # Test basic mapping of litteral labels
        for ds in [Dataset(samples=samples, labels=labels_l, labels_map=od),
                   # Figure out mapping
                   Dataset(samples=samples, labels=labels_l, labels_map=True)]:
            self.failUnless(N.all(ds.labels == [0, 1, 0]))
            self.failUnless(ds.labels_map == od)
            ds_ = ds[1]
            self.failUnless(ds_.labels_map == od,
                msg='selectSamples should provide full mapping preserved')

        # We should complaint about insufficient mapping
        self.failUnlessRaises(ValueError, Dataset, samples=samples,
            labels=labels_l, labels_map = {'apple':0})

        # Conformance to older behavior -- if labels are given in
        # strings, no mapping occur by default
        ds2 = Dataset(samples=samples, labels=labels_l)
        self.failUnlessEqual(ds2.labels_map, None)

        # We should label numerical labels if it was requested:
        od3 = {1:100, 2:101, 3:100}
        ds3 = Dataset(samples=samples, labels=[1,2,3],
                      labels_map = od3)
        self.failUnlessEqual(ds3.labels_map, od3)
        self.failUnless(N.all(ds3.labels == [100, 101, 100]))

        ds3_ = ds3[1]
        self.failUnlessEqual(ds3.labels_map, od3)


def suite():
    return unittest.makeSuite(DatasetTests)


if __name__ == '__main__':
    import runner

