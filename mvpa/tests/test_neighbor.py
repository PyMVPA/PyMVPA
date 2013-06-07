# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA metrics"""


from mvpa.mappers.metric import *
from mvpa.clfs.distance import *
import unittest
import numpy as N

class MetricTests(unittest.TestCase):
    """Basic tests for metrics: neighbors etc
    """

    def testDistances(self):
        a = N.array([3,8])
        b = N.array([6,4])
        # test distances or yarik recalls unit testing ;)
        self.failUnless( cartesianDistance(a, b) == 5.0 )
        self.failUnless( manhattenDistance(a, b) == 7 )
        self.failUnless( absminDistance(a, b) == 4 )


    def testDescreteMetric(self):
        """Descrete metrics tests -- elementsizes etc"""
        # who said that we will not use FSL's data
        # with negative dimensions? :-)
        elsize = [-2.5, 1.5]
        distance = 3

        # use default function
        metric = DescreteMetric(elsize)

        # simple check
        target = N.array([ [1, 2], [2, 1], [2, 2], [2, 3], [3, 2] ])
        self.failUnless( (metric.getNeighbors([2, 2], 2.6) == target).all())

        # a bit longer one... not sure what for
        for point in metric.getNeighbor([2, 2], distance):
            self.failUnless( cartesianDistance(point, [2,2]) <= distance)

        # use manhattenDistance function
        metric = DescreteMetric(elsize, manhattenDistance)
        for point in metric.getNeighbor([2, 2], distance):
            self.failUnless( manhattenDistance(point, [2, 2]) <= distance)

        metric.elementsize = [10, 1.5]
        """We can reassign element size as a whole"""
        self.failUnless((metric.elementsize == [10, 1.5]).all())

        try:
            metric.elementsize[1] = 1
            self.fail(
                msg="We should not be able to reassign parts of elementsize")
        except (ValueError, RuntimeError):
            pass

        self.failUnless((metric.getNeighbors([2, 2], 2.6) ==
                         [t for t in target if t[0]==2]).all())
        """Check if new elementsize is in effect for getNeighbors"""


    def testDescreteMetricCompatMask(self):
        """Test ability to provide compatmask
        """
        # let's play fMRI: 3x3x3.3 mm and 2s TR, but in NIfTI we have it
        # reversed
        esize = [2, 3.3, 3,  3]
        # only the last three axis are spatial ones and compatible in terms
        # of a meaningful distance among them.
        metric = DescreteMetric(esize, compatmask=[0, 1, 1, 1])

        # neighbors in compat space will be simply propagated along remaining
        # dimensions (somewhat like a backprojection from the subspace into the
        # original space)
        self.failUnless((metric.getNeighbors([0]*4, 1) ==
                        [[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]).all())
        # test different radius in compat and in remaining space
        # in this case usual spatial neighborhood, but no temporal
        self.failUnless((metric.getNeighbors([0]*4, [0, 4, 4, 4]) ==
            [[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0,0,0,0],
             [0,0,0,1],[0,0,1,0],[0,1,0,0]]).all())

        # no spatial but temporal neighborhood
        self.failUnless((metric.getNeighbors([0]*4, [2, 0, 0, 0]) ==
            [[-1,0,0,0],[0,0,0,0],[1,0,0,0]]).all())

        # check axis scaling in non-compat space
        self.failUnless(len(metric.getNeighbors([0]*4, [7.9, 0, 0, 0])) == 9)

        # check if we can modify compatmask and still perform fine
        old_filter = metric.filter_coord.copy()
        cm = metric.compatmask
        cm[0] = 1
        metric.compatmask = cm
        self.failUnless((metric.compatmask == [True]*4).all())
        self.failUnless((metric.getNeighbors([0]*4, 3) ==
                         [[-1,  0,  0,  0], [ 0,  0, -1,  0], [ 0,  0,  0, -1],
                          [ 0,  0,  0,  0], [ 0,  0,  0,  1], [ 0,  0,  1,  0],
                          [ 1,  0,  0,  0]]).all())
        self.failUnless(N.any(old_filter != metric.filter_coord))

    def testGetNeighbors(self):
        """Test if generator getNeighbor and method getNeighbors
        return the right thing"""

        class BM(Metric):
            """ Class which overrides only getNeighbor
            """
            def getNeighbor(self):
                for n in [4, 5, 6]: yield n

        class CM(Metric):
            """ Class which overrides only getNeighbor
            """
            def getNeighbors(self):
                return [1, 2, 3]

        b = BM()
        self.failUnless(b.getNeighbors() == [4, 5, 6])
        c = CM()
        self.failUnless([ x for x in c.getNeighbor()] == [1, 2, 3])


def suite():
    """Create the suite -- pylint shut up"""
    return unittest.makeSuite(MetricTests)


if __name__ == '__main__':
    import runner

