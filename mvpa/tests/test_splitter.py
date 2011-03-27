# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA pattern handling"""

import unittest
import numpy as np

from mvpa.testing.datasets import datasets
from mvpa.base.node import ChainNode
from mvpa.datasets.base import dataset_wizard, Dataset
from mvpa.generators.partition import NFoldPartitioner
from mvpa.mappers.slicing import StripBoundariesSamples
from mvpa.generators.splitters import Splitter
from mvpa.generators.partition import *

from mvpa.testing.tools import ok_, assert_array_equal, assert_true, \
        assert_false, assert_equal

class SplitterTests(unittest.TestCase):

    def setUp(self):
        self.data = dataset_wizard(np.random.normal(size=(100,10)),
                            targets=[ i%4 for i in range(100) ],
                            chunks=[ i/10 for i in range(100)])


    def test_simplest_cv_pat_gen(self):
        # create the generator
        nfs = NFoldPartitioner(cvtype=1)
        spl = Splitter(attr='partitions')
        # now get the xval pattern sets One-Fold CV)
        xvpat = [ list(spl.generate(p)) for p in nfs.generate(self.data) ]

        self.failUnless( len(xvpat) == 10 )

        for i,p in enumerate(xvpat):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 90 )
            self.failUnless( p[1].nsamples == 10 )
            self.failUnless( p[1].chunks[0] == i )


    def test_odd_even_split(self):
        oes = OddEvenPartitioner()
        spl = Splitter(attr='partitions')

        splits = [ list(spl.generate(p)) for p in oes.generate(self.data) ]

        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        assert_array_equal(splits[0][1].sa['chunks'].unique, [1, 3, 5, 7, 9])
        assert_array_equal(splits[0][0].sa['chunks'].unique, [0, 2, 4, 6, 8])
        assert_array_equal(splits[1][0].sa['chunks'].unique, [1, 3, 5, 7, 9])
        assert_array_equal(splits[1][1].sa['chunks'].unique, [0, 2, 4, 6, 8])

        # check if it works on pure odd and even chunk ids
        moresplits = [ list(spl.generate(p)) for p in oes.generate(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)


    def test_half_split(self):
        hs = HalfPartitioner()
        spl = Splitter(attr='partitions')

        splits = [ list(spl.generate(p)) for p in hs.generate(self.data) ]

        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        assert_array_equal(splits[0][1].sa['chunks'].unique, [0, 1, 2, 3, 4])
        assert_array_equal(splits[0][0].sa['chunks'].unique, [5, 6, 7, 8, 9])
        assert_array_equal(splits[1][1].sa['chunks'].unique, [5, 6, 7, 8, 9])
        assert_array_equal(splits[1][0].sa['chunks'].unique, [0, 1, 2, 3, 4])

        # check if it works on pure odd and even chunk ids
        moresplits = [ list(spl.generate(p)) for p in hs.generate(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)

    def test_n_group_split(self):
        """Test NGroupSplitter alongside with the reversal of the
        order of spit out datasets
        """
        # Test 2 groups like HalfSplitter first
        hs = NGroupPartitioner(2)

        for isreversed, splitter in enumerate((hs, hs)):
            if isreversed:
                spl = Splitter(attr='partitions', reverse=True)
            else:
                spl = Splitter(attr='partitions')
            splits = [ list(spl.generate(p)) for p in hs.generate(self.data) ]
            self.failUnless(len(splits) == 2)

            for i, p in enumerate(splits):
                self.failUnless( len(p) == 2 )
                self.failUnless( p[0].nsamples == 50 )
                self.failUnless( p[1].nsamples == 50 )

            assert_array_equal(splits[0][1-isreversed].sa['chunks'].unique,
                               [0, 1, 2, 3, 4])
            assert_array_equal(splits[0][isreversed].sa['chunks'].unique,
                               [5, 6, 7, 8, 9])
            assert_array_equal(splits[1][1-isreversed].sa['chunks'].unique,
                               [5, 6, 7, 8, 9])
            assert_array_equal(splits[1][isreversed].sa['chunks'].unique,
                               [0, 1, 2, 3, 4])

        # check if it works on pure odd and even chunk ids
        moresplits = [ list(spl.generate(p)) for p in hs.generate(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)

        # now test more groups
        s5 = NGroupPartitioner(5)

        # get the splits
        for isreversed, s5splitter in enumerate((s5, s5)):
            if isreversed:
                spl = Splitter(attr='partitions', reverse=True)
            else:
                spl = Splitter(attr='partitions')
            splits = [ list(spl.generate(p)) for p in s5splitter.generate(self.data) ]

            # must have 10 splits
            self.failUnless(len(splits) == 5)

            # check split content
            assert_array_equal(splits[0][1-isreversed].sa['chunks'].unique,
                               [0, 1])
            assert_array_equal(splits[0][isreversed].sa['chunks'].unique,
                               [2, 3, 4, 5, 6, 7, 8, 9])
            assert_array_equal(splits[1][1-isreversed].sa['chunks'].unique,
                               [2, 3])
            assert_array_equal(splits[1][isreversed].sa['chunks'].unique,
                               [0, 1, 4, 5, 6, 7, 8, 9])
            # ...
            assert_array_equal(splits[4][1-isreversed].sa['chunks'].unique,
                               [8, 9])
            assert_array_equal(splits[4][isreversed].sa['chunks'].unique,
                               [0, 1, 2, 3, 4, 5, 6, 7])


        # Test for too many groups
        def splitcall(spl, dat):
            return list(spl.generate(dat))
        s20 = NGroupPartitioner(20)
        self.assertRaises(ValueError,splitcall,s20,self.data)

    def test_custom_split(self):
        #simulate half splitter
        hs = CustomPartitioner([(None,[0,1,2,3,4]),(None,[5,6,7,8,9])])
        spl = Splitter(attr='partitions')
        splits = [ list(spl.generate(p)) for p in hs.generate(self.data) ]
        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        assert_array_equal(splits[0][1].sa['chunks'].unique, [0, 1, 2, 3, 4])
        assert_array_equal(splits[0][0].sa['chunks'].unique, [5, 6, 7, 8, 9])
        assert_array_equal(splits[1][1].sa['chunks'].unique, [5, 6, 7, 8, 9])
        assert_array_equal(splits[1][0].sa['chunks'].unique, [0, 1, 2, 3, 4])


        # check fully customized split with working and validation set specified
        cs = CustomPartitioner([([0,3,4],[5,9])])
        # we want to discared the unselected partition of the data, hence attr_value
        # these two splitters should do exactly the same thing
        splitters = (Splitter(attr='partitions', attr_values=[1,2]),
                     Splitter(attr='partitions', ignore_values=(0,)))
        for spl in splitters:
            splits = [ list(spl.generate(p)) for p in cs.generate(self.data) ]
            self.failUnless(len(splits) == 1)

            for i,p in enumerate(splits):
                self.failUnless( len(p) == 2 )
                self.failUnless( p[0].nsamples == 30 )
                self.failUnless( p[1].nsamples == 20 )

            self.failUnless((splits[0][1].sa['chunks'].unique == [5, 9]).all())
            self.failUnless((splits[0][0].sa['chunks'].unique == [0, 3, 4]).all())


    def test_label_splitter(self):
        oes = OddEvenPartitioner(attr='targets')
        spl = Splitter(attr='partitions')

        splits = [ list(spl.generate(p)) for p in oes.generate(self.data) ]

        assert_array_equal(splits[0][0].sa['targets'].unique, [0,2])
        assert_array_equal(splits[0][1].sa['targets'].unique, [1,3])
        assert_array_equal(splits[1][0].sa['targets'].unique, [1,3])
        assert_array_equal(splits[1][1].sa['targets'].unique, [0,2])


    def test_counted_splitting(self):
        spl = Splitter(attr='partitions')
        # count > #chunks, should result in 10 splits
        nchunks = len(self.data.sa['chunks'].unique)
        for strategy in Partitioner._STRATEGIES:
            for count, target in [ (nchunks*2, nchunks),
                                   (nchunks, nchunks),
                                   (nchunks-1, nchunks-1),
                                   (3, 3),
                                   (0, 0),
                                   (1, 1)
                                   ]:
                nfs = NFoldPartitioner(cvtype=1, count=count,
                                       selection_strategy=strategy)
                splits = [ list(spl.generate(p)) for p in nfs.generate(self.data) ]
                self.failUnless(len(splits) == target)
                chosenchunks = [int(s[1].uniquechunks) for s in splits]

                # Test if configuration matches as well
                nsplits_cfg = len(nfs.get_partition_specs(self.data))
                self.failUnlessEqual(nsplits_cfg, target)

                # Check if "lastsplit" dsattr was assigned appropriately
                nsplits = len(splits)
                if nsplits > 0:
                    # dummy-proof testing of last split
                    for ds_ in splits[-1]:
                        self.failUnless(ds_.a.lastpartitionset)
                    # test all now
                    for isplit,split in enumerate(splits):
                        for ds_ in split:
                            ds_.a.lastpartitionset == isplit==nsplits-1

                # Check results of different strategies
                if strategy == 'first':
                    self.failUnlessEqual(chosenchunks, range(target))
                elif strategy == 'equidistant':
                    if target == 3:
                        self.failUnlessEqual(chosenchunks, [0, 3, 7])
                elif strategy == 'random':
                    # none is selected twice
                    self.failUnless(len(set(chosenchunks)) == len(chosenchunks))
                    self.failUnless(target == len(chosenchunks))
                else:
                    raise RuntimeError, "Add unittest for strategy %s" \
                          % strategy


    def test_discarded_boundaries(self):
        ds = datasets['hollow']
        # four runs
        ds.sa['chunks'] = np.repeat(np.arange(4), 10)
        # do odd even splitting for lots of boundaries in few splits
        part = ChainNode([OddEvenPartitioner(),
                          StripBoundariesSamples('chunks', 1, 2)])

        parts = [d.samples.sid for d in part.generate(ds)]

        # both dataset should have the same samples, because the boundaries are
        # identical and the same sample should be stripped
        assert_array_equal(parts[0], parts[1])

        # we strip 3 samples per boundary
        assert_equal(len(parts[0]), len(ds) - (3 * 3))

        for i in [9, 10, 11, 19, 20, 21, 29, 30, 31]:
            assert_false(i in parts[0])


    def test_slicing(self):
        hs = HalfPartitioner()
        spl = Splitter(attr='partitions')
        splits = list(hs.generate(self.data))
        for s in splits:
            # partitioned dataset shared the data
            assert_true(s.samples.base is self.data.samples)
        splits = [ list(spl.generate(p)) for p in hs.generate(self.data) ]
        for s in splits:
            # we get slicing all the time
            assert_true(s[0].samples.base.base is self.data.samples)
            assert_true(s[1].samples.base.base is self.data.samples)
        spl = Splitter(attr='partitions', noslicing=True)
        splits = [ list(spl.generate(p)) for p in hs.generate(self.data) ]
        for s in splits:
            # we no slicing at all
            assert_false(s[0].samples.base is self.data.samples)
            assert_false(s[1].samples.base is self.data.samples)
        nfs = NFoldPartitioner()
        spl = Splitter(attr='partitions')
        splits = [ list(spl.generate(p)) for p in nfs.generate(self.data) ]
        for i, s in enumerate(splits):
            # training only first and last split
            if i == 0 or i == len(splits) - 1:
                assert_true(s[0].samples.base.base is self.data.samples)
            else:
                assert_true(s[0].samples.base is None)
            # we get slicing all the time
            assert_true(s[1].samples.base.base is self.data.samples)
        step_ds = Dataset(np.random.randn(20,2),
                          sa={'chunks': np.tile([0,1], 10)})
        oes = OddEvenPartitioner()
        spl = Splitter(attr='partitions')
        splits = list(oes.generate(step_ds))
        for s in splits:
            # partitioned dataset shared the data
            assert_true(s.samples.base is step_ds.samples)
        splits = [ list(spl.generate(p)) for p in oes.generate(step_ds) ]
        assert_equal(len(splits), 2)
        for s in splits:
            # we get slicing all the time
            assert_true(s[0].samples.base.base is step_ds.samples)
            assert_true(s[1].samples.base.base is step_ds.samples)


def suite():
    return unittest.makeSuite(SplitterTests)


if __name__ == '__main__':
    import runner

