# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA pattern handling"""

from mvpa.datasets.masked import MaskedDataset
from mvpa.datasets.splitters import NFoldSplitter, OddEvenSplitter, \
                                   NoneSplitter, HalfSplitter, \
                                   CustomSplitter, NGroupSplitter
import unittest
import numpy as N


class SplitterTests(unittest.TestCase):

    def setUp(self):
        self.data = \
            MaskedDataset(samples=N.random.normal(size=(100,10)),
                           labels=[ i%4 for i in range(100) ],
                            chunks=[ i/10 for i in range(100)])


    def testSimplestCVPatGen(self):
        # create the generator
        nfs = NFoldSplitter(cvtype=1)

        # now get the xval pattern sets One-Fold CV)
        xvpat = [ (train, test) for (train,test) in nfs(self.data) ]

        self.failUnless( len(xvpat) == 10 )

        for i,p in enumerate(xvpat):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 90 )
            self.failUnless( p[1].nsamples == 10 )
            self.failUnless( p[1].chunks[0] == i )


    def testOddEvenSplit(self):
        oes = OddEvenSplitter()

        splits = [ (train, test) for (train, test) in oes(self.data) ]

        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        self.failUnless((splits[0][1].uniquechunks == [1, 3, 5, 7, 9]).all())
        self.failUnless((splits[0][0].uniquechunks == [0, 2, 4, 6, 8]).all())
        self.failUnless((splits[1][0].uniquechunks == [1, 3, 5, 7, 9]).all())
        self.failUnless((splits[1][1].uniquechunks == [0, 2, 4, 6, 8]).all())

        # check if it works on pure odd and even chunk ids
        moresplits = [ (train, test) for (train, test) in oes(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)


    def testHalfSplit(self):
        hs = HalfSplitter()

        splits = [ (train, test) for (train, test) in hs(self.data) ]

        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        self.failUnless((splits[0][1].uniquechunks == [0, 1, 2, 3, 4]).all())
        self.failUnless((splits[0][0].uniquechunks == [5, 6, 7, 8, 9]).all())
        self.failUnless((splits[1][1].uniquechunks == [5, 6, 7, 8, 9]).all())
        self.failUnless((splits[1][0].uniquechunks == [0, 1, 2, 3, 4]).all())

        # check if it works on pure odd and even chunk ids
        moresplits = [ (train, test) for (train, test) in hs(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)

    def testNGroupSplit(self):
        """Test NGroupSplitter alongside with the reversal of the
        order of spit out datasets
        """
        # Test 2 groups like HalfSplitter first
        hs = NGroupSplitter(2)
        hs_reversed = NGroupSplitter(2, reverse=True)

        for isreversed, splitter in enumerate((hs, hs_reversed)):
            splits = list(splitter(self.data))
            self.failUnless(len(splits) == 2)

            for i, p in enumerate(splits):
                self.failUnless( len(p) == 2 )
                self.failUnless( p[0].nsamples == 50 )
                self.failUnless( p[1].nsamples == 50 )

            self.failUnless((splits[0][1-isreversed].uniquechunks == [0, 1, 2, 3, 4]).all())
            self.failUnless((splits[0][isreversed].uniquechunks == [5, 6, 7, 8, 9]).all())
            self.failUnless((splits[1][1-isreversed].uniquechunks == [5, 6, 7, 8, 9]).all())
            self.failUnless((splits[1][isreversed].uniquechunks == [0, 1, 2, 3, 4]).all())

        # check if it works on pure odd and even chunk ids
        moresplits = list(hs(splits[0][0]))

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)

        # now test more groups
        s5 = NGroupSplitter(5)
        s5_reversed = NGroupSplitter(5, reverse=True)

        # get the splits
        for isreversed, s5splitter in enumerate((s5, s5_reversed)):
            splits = list(s5splitter(self.data))

            # must have 10 splits
            self.failUnless(len(splits) == 5)

            # check split content
            self.failUnless((splits[0][1-isreversed].uniquechunks == [0, 1]).all())
            self.failUnless((splits[0][isreversed].uniquechunks == [2, 3, 4, 5, 6, 7, 8, 9]).all())
            self.failUnless((splits[1][1-isreversed].uniquechunks == [2, 3]).all())
            self.failUnless((splits[1][isreversed].uniquechunks == [0, 1, 4, 5, 6, 7, 8, 9]).all())
            # ...
            self.failUnless((splits[4][1-isreversed].uniquechunks == [8, 9]).all())
            self.failUnless((splits[4][isreversed].uniquechunks == [0, 1, 2, 3, 4, 5, 6, 7]).all())


        # Test for too many groups
        def splitcall(spl, dat):
            return [ (train, test) for (train, test) in spl(dat) ]
        s20 = NGroupSplitter(20)
        self.assertRaises(ValueError,splitcall,s20,self.data)

    def testCustomSplit(self):
        #simulate half splitter
        hs = CustomSplitter([(None,[0,1,2,3,4]),(None,[5,6,7,8,9])])
        splits = list(hs(self.data))
        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        self.failUnless((splits[0][1].uniquechunks == [0, 1, 2, 3, 4]).all())
        self.failUnless((splits[0][0].uniquechunks == [5, 6, 7, 8, 9]).all())
        self.failUnless((splits[1][1].uniquechunks == [5, 6, 7, 8, 9]).all())
        self.failUnless((splits[1][0].uniquechunks == [0, 1, 2, 3, 4]).all())


        # check fully customized split with working and validation set specified
        cs = CustomSplitter([([0,3,4],[5,9])])
        splits = list(cs(self.data))
        self.failUnless(len(splits) == 1)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 30 )
            self.failUnless( p[1].nsamples == 20 )

        self.failUnless((splits[0][1].uniquechunks == [5, 9]).all())
        self.failUnless((splits[0][0].uniquechunks == [0, 3, 4]).all())

        # full test with additional sampling and 3 datasets per split
        cs = CustomSplitter([([0,3,4],[5,9],[2])],
                            nperlabel=[3,4,1],
                            nrunspersplit=3)
        splits = list(cs(self.data))
        self.failUnless(len(splits) == 3)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 3 )
            self.failUnless( p[0].nsamples == 12 )
            self.failUnless( p[1].nsamples == 16 )
            self.failUnless( p[2].nsamples == 4 )

        # lets test selection of samples by ratio and combined with
        # other ways
        cs = CustomSplitter([([0,3,4],[5,9],[2])],
                            nperlabel=[[0.3, 0.6, 1.0, 0.5],
                                       0.5,
                                       'all'],
                            nrunspersplit=3)
        csall = CustomSplitter([([0,3,4],[5,9],[2])],
                               nrunspersplit=3)
        # lets craft simpler dataset
        #ds = Dataset(samples=N.arange(12), labels=[1]*6+[2]*6, chunks=1)
        splits = list(cs(self.data))
        splitsall = list(csall(self.data))

        self.failUnless(len(splits) == 3)
        ul = self.data.uniquelabels

        self.failUnless(((N.array(splitsall[0][0].samplesperlabel.values())
                          *[0.3, 0.6, 1.0, 0.5]).round().astype(int) ==
                         N.array(splits[0][0].samplesperlabel.values())).all())

        self.failUnless(((N.array(splitsall[0][1].samplesperlabel.values())*0.5
                          ).round().astype(int) ==
                         N.array(splits[0][1].samplesperlabel.values())).all())

        self.failUnless((N.array(splitsall[0][2].samplesperlabel.values()) ==
                         N.array(splits[0][2].samplesperlabel.values())).all())


    def testNoneSplitter(self):
        nos = NoneSplitter()
        splits = [ (train, test) for (train, test) in nos(self.data) ]
        self.failUnless(len(splits) == 1)
        self.failUnless(splits[0][0] == None)
        self.failUnless(splits[0][1].nsamples == 100)

        nos = NoneSplitter(mode='first')
        splits = [ (train, test) for (train, test) in nos(self.data) ]
        self.failUnless(len(splits) == 1)
        self.failUnless(splits[0][1] == None)
        self.failUnless(splits[0][0].nsamples == 100)


        # test sampling tools
        # specified value
        nos = NoneSplitter(nrunspersplit=3,
                           nperlabel=10)
        splits = [ (train, test) for (train, test) in nos(self.data) ]

        self.failUnless(len(splits) == 3)
        for split in splits:
            self.failUnless(split[0] == None)
            self.failUnless(split[1].nsamples == 40)
            self.failUnless(split[1].samplesperlabel.values() == [10,10,10,10])

        # auto-determined
        nos = NoneSplitter(nrunspersplit=3,
                           nperlabel='equal')
        splits = [ (train, test) for (train, test) in nos(self.data) ]

        self.failUnless(len(splits) == 3)
        for split in splits:
            self.failUnless(split[0] == None)
            self.failUnless(split[1].nsamples == 100)
            self.failUnless(split[1].samplesperlabel.values() == [25,25,25,25])


    def testLabelSplitter(self):
        oes = OddEvenSplitter(attr='labels')

        splits = [ (first, second) for (first, second) in oes(self.data) ]

        self.failUnless((splits[0][0].uniquelabels == [0,2]).all())
        self.failUnless((splits[0][1].uniquelabels == [1,3]).all())
        self.failUnless((splits[1][0].uniquelabels == [1,3]).all())
        self.failUnless((splits[1][1].uniquelabels == [0,2]).all())


    def testCountedSplitting(self):
        # count > #chunks, should result in 10 splits
        nchunks = len(self.data.uniquechunks)
        for strategy in NFoldSplitter._STRATEGIES:
            for count, target in [ (nchunks*2, nchunks),
                                   (nchunks, nchunks),
                                   (nchunks-1, nchunks-1),
                                   (3, 3),
                                   (0, 0),
                                   (1, 1)
                                   ]:
                nfs = NFoldSplitter(cvtype=1, count=count, strategy=strategy)
                splits = [ (train, test) for (train,test) in nfs(self.data) ]
                self.failUnless(len(splits) == target)
                chosenchunks = [int(s[1].uniquechunks) for s in splits]
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


    def testDiscardedBoundaries(self):
        splitters = [NFoldSplitter(),
                     NFoldSplitter(discard_boundary=(0,1)), # discard testing
                     NFoldSplitter(discard_boundary=(1,0)), # discard training
                     NFoldSplitter(discard_boundary=(2,0)), # discard 2 from training
                     NFoldSplitter(discard_boundary=1),     # discard from both
                     OddEvenSplitter(discard_boundary=(1,0)),
                     OddEvenSplitter(discard_boundary=(0,1)),
                     HalfSplitter(discard_boundary=(1,0)),
                     ]

        split_sets = [list(s(self.data)) for s in splitters]
        counts = [[(len(s[0].chunks), len(s[1].chunks)) for s in split_set]
                  for split_set in split_sets]

        nodiscard_tr = [c[0] for c in counts[0]]
        nodiscard_te = [c[1] for c in counts[0]]

        # Discarding in testing:
        self.failUnless(nodiscard_tr == [c[0] for c in counts[1]])
        self.failUnless(nodiscard_te[1:-1] == [c[1] + 2 for c in counts[1][1:-1]])
        # at the beginning/end chunks, just a single element
        self.failUnless(nodiscard_te[0] == counts[1][0][1] + 1)
        self.failUnless(nodiscard_te[-1] == counts[1][-1][1] + 1)

        # Discarding in training
        for d in [1,2]:
            self.failUnless(nodiscard_te == [c[1] for c in counts[1+d]])
            self.failUnless(nodiscard_tr[0] == counts[1+d][0][0] + d)
            self.failUnless(nodiscard_tr[-1] == counts[1+d][-1][0] + d)
            self.failUnless(nodiscard_tr[1:-1] == [c[0] + d*2
                                                   for c in counts[1+d][1:-1]])

        # Discarding in both -- should be eq min from counts[1] and [2]
        counts_min = [(min(c1[0], c2[0]), min(c1[1], c2[1]))
                      for c1,c2 in zip(counts[1], counts[2])]
        self.failUnless(counts_min == counts[4])

        # TODO: test all those odd/even etc splitters... YOH: did
        # visually... looks ok;)
        #for count in counts[5:]:
        #    print count


def suite():
    return unittest.makeSuite(SplitterTests)


if __name__ == '__main__':
    import runner

