# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dense array mapper"""


from mvpa.mappers.array import DenseArrayMapper
from mvpa.mappers.metric import *
import unittest
import numpy as N

class DenseArrayMapperTests(unittest.TestCase):

    def testForwardDenseArrayMapper(self):
        mask = N.ones((3,2))
        map_ = DenseArrayMapper(mask)

        # test shape reports
        self.failUnless( map_.nfeatures == 6 )

        # test 1sample mapping
        self.failUnless( ( map_.forward( N.arange(6).reshape(3,2) ) \
                           == [0,1,2,3,4,5]).all() )

        # test 4sample mapping
        foursample = map_.forward( N.arange(24).reshape(4,3,2)) 
        self.failUnless( ( foursample \
                           == [[0,1,2,3,4,5],
                               [6,7,8,9,10,11],
                               [12,13,14,15,16,17],
                               [18,19,20,21,22,23]]).all() )

        # check incomplete masks
        mask[1,1] = 0
        map_ = DenseArrayMapper(mask)
        self.failUnless( map_.nfeatures == 5 )
        self.failUnless( ( map_.forward( N.arange(6).reshape(3,2) ) \
                           == [0,1,2,4,5]).all() )

        # check that it doesn't accept wrong dataspace
        self.failUnlessRaises( ValueError,
                               map_.forward,
                               N.arange(4).reshape(2,2) )

        # check fail if neither mask nor shape
        self.failUnlessRaises(ValueError, DenseArrayMapper)

        # check that a full mask is automatically created when providing shape
        m = DenseArrayMapper(shape=(2, 3, 4))
        mp = m.forward(N.arange(24).reshape(2, 3, 4))
        self.failUnless((mp == N.arange(24)).all())


    def testReverseDenseArrayMapper(self):
        mask = N.ones((3,2))
        mask[1,1] = 0
        map_ = DenseArrayMapper(mask)

        rmapped = map_.reverse(N.arange(1,6))
        self.failUnless( rmapped.shape == (3,2) )
        self.failUnless( rmapped[1,1] == 0 )
        self.failUnless( rmapped[2,1] == 5 )


        # check that it doesn't accept wrong dataspace
        self.failUnlessRaises( ValueError,
                               map_,
                               N.arange(6))

        rmapped2 = map_.reverse(N.arange(1,11).reshape(2,5))
        self.failUnless( rmapped2.shape == (2,3,2) )
        self.failUnless( rmapped2[0,1,1] == 0 )
        self.failUnless( rmapped2[1,1,1] == 0 )
        self.failUnless( rmapped2[0,2,1] == 5 )
        self.failUnless( rmapped2[1,2,1] == 10 )


    def testDenseArrayMapperMetrics(self):
        """ Test DenseArrayMapperMetric
        """
        mask = N.ones((3,2))
        mask[1,1] = 0

        # take space with non-square elements
        neighborFinder = DescreteMetric([0.5, 2])
        map_ = DenseArrayMapper(mask, neighborFinder)

        # test getNeighbors
        # now it returns list of arrays
        #target = [N.array([0, 0]), N.array([0, 1]),
        #          N.array([1, 0]), N.array([2, 0])]
        #result = map_.getNeighborIn([0, 0], 2)
        #self.failUnless(N.array(map(lambda x,y:(x==y).all(), result, target)).all())

        # check by providing outId
        target = [0,1,2,3]
        result = map_.getNeighbors(0, 2.1)
        self.failUnless( result == target )

        map__ = DenseArrayMapper(mask, elementsize=[0.5, 2])
        self.failUnless( map__.getNeighbors(0, 2.1) == target,
                         msg="DenseArrayMapper must accept elementsize parameter and set" +
                         " DescreteMetric accordingly")

        self.failUnlessRaises(ValueError, DenseArrayMapper,
                                          mask, elementsize=[0.5]*3)
        """DenseArrayMapper must raise exception when not appropriatly sized
        elementsize was provided"""



    def testMapperAliases(self):
        mm=DenseArrayMapper(N.ones((3,4,2)))
        # We decided to don't have alias for reverse
        #self.failUnless((mm(N.arange(24)) == mm.reverse(N.arange(24))).all())
        self.failUnless((mm(N.ones((3,4,2))) \
                        == mm.forward(N.ones((3,4,2)))).all())


    def testGetInOutIdBehaviour(self):
        mask=N.zeros((3,4,2))
        mask[0,0,1]=1
        mask[2,1,0]=1
        mask[0,3,1]=1

        mm=DenseArrayMapper(mask)

        self.failUnless(mm.nfeatures==3)

        # 'In' 
        self.failUnless((mm.getInIds() \
                         == N.array([[0, 0, 1],[0, 3, 1],[2, 1, 0]])).all())
        self.failUnless((mm.getInId(1) == [0,3,1]).all())
        # called with list gives nonzero() like output
        self.failUnless((mm.getInId(range(mm.nfeatures)) \
                         == mm.getInIds().T).all())

        # 'Out'
        self.failUnlessRaises( ValueError,
                               mm.getOutId,
                               (0,0,0))
        self.failUnless(mm.getOutId((0,0,1)) == 0
                        and mm.getOutId((0,3,1)) == 1
                        and mm.getOutId((2,1,0)) == 2)


    def testSelects(self):
        mask = N.ones((3,2))
        mask[1,1] = 0
        mask0 = mask.copy()
        data = N.arange(6).reshape(mask.shape)
        map_ = DenseArrayMapper(mask)

        # check if any exception is thrown if we get
        # out of the outIds
        self.failUnlessRaises(IndexError, map_.selectOut, [0,1,2,6])

        # remove 1,2
        map_.selectOut([0,3,4])
        self.failUnless((map_.forward(data)==[0, 4, 5]).all())
        # remove 1 more
        map_.selectOut([0,2])
        self.failUnless((map_.forward(data)==[0, 5]).all())

        # check if original mask wasn't perturbed
        self.failUnless((mask == mask0).all())

        # do the same but using discardOut
        map_ = DenseArrayMapper(mask)
        map_.discardOut([1,2])
        self.failUnless((map_.forward(data)==[0, 4, 5]).all())
        map_.discardOut([1])
        self.failUnless((map_.forward(data)==[0, 5]).all())

        # check if original mask wasn't perturbed
        self.failUnless((mask == mask0).all())


    def _testSelectReOrder(self):
        """
        Test is desabled for now since if order is incorrect in
        __debug__ we just spit out a warning - no exception
        """
        mask = N.ones((3,3))
        mask[1,1] = 0

        data = N.arange(9).reshape(mask.shape)
        map_ = DenseArrayMapper(mask)
        oldneighbors = map_.forward(data)[map_.getNeighbors(0, radius=2)]

        # just do place changes
        # by default - we don't sort/check order so it would screw things
        # up
        map_.selectOut([7, 1, 2, 3, 4, 5, 6, 0])
        # we check if an item new outId==7 still has proper neighbors
        newneighbors = map_.forward(data)[map_.getNeighbors(7, radius=2)]
        self.failUnless( (oldneighbors != newneighbors ).any())

# disable since selectOut does not have 'sort' anymore
#    def testSelectOrder(self):
#        """
#        Test if changing the order by doing selectOut preserves
#        neighborhood information -- but also apply sort in difference
#        to testSelectReOrder
#        """
#        mask = N.ones((3,3))
#        mask[1,1] = 0
#
#        data = N.arange(9).reshape(mask.shape)
#        map_ = DenseArrayMapper(mask)
#        oldneighbors = map_.forward(data)[map_.getNeighbors(0, radius=2)]
#
#        map_ = DenseArrayMapper(mask)
#        map_.selectOut([7, 1, 2, 3, 4, 5, 6, 0], sort=True)
#        # we check if an item new outId==0 still has proper neighbors
#        newneighbors = map_.forward(data)[map_.getNeighbors(0, radius=2)]
#        self.failUnless( (oldneighbors == newneighbors ).all())


def suite():
    return unittest.makeSuite(DenseArrayMapperTests)


if __name__ == '__main__':
    import runner

