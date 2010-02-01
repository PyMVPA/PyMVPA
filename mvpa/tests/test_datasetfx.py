# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA miscelaneouse functions operating on datasets"""

import unittest

import numpy as N

from mvpa.base import externals
from mvpa.datasets import Dataset
from mvpa.datasets.miscfx import removeInvariantFeatures, coarsenChunks, \
     SequenceStats

from mvpa.misc.data_generators import normalFeatureDataset

class MiscDatasetFxTests(unittest.TestCase):

    def testInvarFeaturesRemoval(self):
        r = N.random.normal(size=(3,1))
        ds = Dataset(samples=N.hstack((N.zeros((3,2)), r)),
                     labels=1)

        self.failUnless(ds.nfeatures == 3)

        dsc = removeInvariantFeatures(ds)

        self.failUnless(dsc.nfeatures == 1)
        self.failUnless((dsc.samples == r).all())


    def testCoarsenChunks(self):
        """Just basic testing for now"""
        chunks = [1,1,2,2,3,3,4,4]
        ds = Dataset(samples=N.arange(len(chunks)).reshape(
            (len(chunks),1)), labels=[1]*8, chunks=chunks)
        coarsenChunks(ds, nchunks=2)
        chunks1 = coarsenChunks(chunks, nchunks=2)
        self.failUnless((chunks1 == ds.chunks).all())
        self.failUnless((chunks1 == N.asarray([0,0,0,0,1,1,1,1])).all())

        ds2 = Dataset(samples=N.arange(len(chunks)).reshape(
            (len(chunks),1)), labels=[1]*8)
        coarsenChunks(ds2, nchunks=2)
        self.failUnless((chunks1 == ds.chunks).all())

    def testBinds(self):
        ds = normalFeatureDataset()
        ds_data = ds.samples.copy()
        ds_chunks = ds.chunks.copy()
        self.failUnless(N.all(ds.samples == ds_data)) # sanity check

        funcs = ['zscore', 'coarsenChunks']
        if externals.exists('scipy'):
            funcs.append('detrend')

        for f in funcs:
            eval('ds.%s()' % f)
            self.failUnless(N.any(ds.samples != ds_data) or
                            N.any(ds.chunks != ds_chunks),
                msg="We should have modified original dataset with %s" % f)
            ds.samples = ds_data.copy()
            ds.chunks = ds_chunks.copy()

        # and some which should just return results
        for f in ['aggregateFeatures', 'removeInvariantFeatures',
                  'getSamplesPerChunkLabel']:
            res = eval('ds.%s()' % f)
            self.failUnless(res is not None,
                msg='We should have got result from function %s' % f)
            self.failUnless(N.all(ds.samples == ds_data),
                msg="Function %s should have not modified original dataset" % f)

    def testSequenceStat(self):
        """Test sequence statistics
        """
        order = 3
        # Close to perfectly balanced one
        sp = N.array([-1,  1,  1, -1,  1, -1, -1,  1, -1, -1, -1,
                      -1,  1, -1,  1, -1,  1,  1,  1, -1,  1,  1,
                      -1, -1, -1,  1,  1,  1,  1,  1, -1], dtype=int)
        rp = SequenceStats(sp, order=order)
        self.failUnlessAlmostEqual(rp['sumabscorr'], 1.0)
        self.failUnlessAlmostEqual(N.max(rp['corrcoef'] * (len(sp)-1) + 1.0), 0.0)

        # Now some random but loong one still binary (boolean)
        sb = (N.random.random_sample((1000,)) >= 0.5)
        rb = SequenceStats(sb, order=order)

        # Now lets do multiclass with literal labels
        s5 = N.random.permutation(['f', 'bu', 'd', 0, 'zz']*200)
        r5 = SequenceStats(s5, order=order)

        # Degenerate one but still should be valid
        s1 = ['aaa']*100
        r1 = SequenceStats(s1, order=order)

        # Generic conformance tests
        for r in (rp, rb, r5, r1):
            ulabels = r['ulabels']
            nlabels = len(r['ulabels'])
            cbcounts = r['cbcounts']
            self.failUnlessEqual(len(cbcounts), order)
            for cb in cbcounts:
                self.failUnlessEqual(N.asarray(cb).shape, (nlabels, nlabels))
            # Check if str works fine
            sr = str(r)
            # TODO: check the content

def suite():
    return unittest.makeSuite(MiscDatasetFxTests)


if __name__ == '__main__':
    import runner

