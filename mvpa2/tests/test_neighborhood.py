# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os

import numpy as np
from numpy import array

from mvpa2.datasets.base import Dataset
import mvpa2.misc.neighborhood as ne
from mvpa2.clfs.distance import *

from mvpa2.testing.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_array_equal
from mvpa2.testing.datasets import datasets

def test_distances():
    a = np.array([3,8])
    b = np.array([6,4])
    # test distances or yarik recalls unit testing ;)
    assert_equal(cartesian_distance(a, b), 5.0)
    assert_equal(manhatten_distance(a, b), 7)
    assert_equal(absmin_distance(a, b), 4)


def test_identity():
    # IdentityNeighborhood() behaves like Sphere(0.5) without all of the
    # computation. Test on a few different coordinates.
    neighborhood = ne.IdentityNeighborhood()
    sphere = ne.Sphere(0.5)
    for center in ((0, 0, 0), (1, 1, 1), (0, 0), (0, )):
        assert_array_equal(neighborhood(center), sphere(center))


def test_sphere():
    # test sphere initialization
    s = ne.Sphere(1)
    center0 = (0, 0, 0)
    center1 = (1, 1, 1)
    assert_equal(len(s(center0)), 7)
    target = array([array([-1,  0,  0]),
              array([ 0, -1,  0]),
              array([ 0,  0, -1]),
              array([0, 0, 0]),
              array([0, 0, 1]),
              array([0, 1, 0]),
              array([1, 0, 0])])
    # test of internals -- no recomputation of increments should be done
    prev_increments = s._increments
    assert_array_equal(s(center0), target)
    ok_(prev_increments is s._increments)
    # query lower dimensionality
    _ = s((0, 0))
    ok_(not prev_increments is s._increments)

    # test Sphere call
    target = [array([0, 1, 1]),
              array([1, 0, 1]),
              array([1, 1, 0]),
              array([1, 1, 1]),
              array([1, 1, 2]),
              array([1, 2, 1]),
              array([2, 1, 1])]
    res = s(center1)
    assert_array_equal(array(res), target)
    # They all should be tuples
    ok_(np.all([isinstance(x, tuple) for x in res]))

    # test for larger diameter
    s = ne.Sphere(4)
    assert_equal(len(s(center1)), 257)

    # test extent keyword
    #s = ne.Sphere(4,extent=(1,1,1))
    #assert_array_equal(array(s((0,0,0))), array([[0,0,0]]))

    # test Errors during initialisation and call
    #assert_raises(ValueError, ne.Sphere, 2)
    #assert_raises(ValueError, ne.Sphere, 1.0)

    # no longer extent available
    assert_raises(TypeError, ne.Sphere, 1, extent=(1))
    assert_raises(TypeError, ne.Sphere, 1, extent=(1.0, 1.0, 1.0))

    s = ne.Sphere(1)
    #assert_raises(ValueError, s, (1))
    if __debug__:
        # No float coordinates allowed for now...
        # XXX might like to change that ;)
        # 
        assert_raises(ValueError, s, (1.0, 1.0, 1.0))


def test_sphere_distance_func():
    # Test some other distance
    se = ne.Sphere(3)
    sm = ne.Sphere(3, distance_func=manhatten_distance)
    rese = se((10, 5))
    resm = sm((10, 5))
    for res in rese, resm:
        # basic test for duplicates (I think we forgotten to test for them)
        ok_(len(res) == len(set(res)))

    # in manhatten distance we should all be no further than 3 "steps" away
    ok_(np.all([np.sum(np.abs(np.array(x) - (10, 5))) <= 3 for x in resm]))
    # in euclidean we are taking shortcuts ;)
    ok_(np.any([np.sum(np.abs(np.array(x) - (10, 5))) > 3 for x in rese]))


def test_sphere_scaled():
    s1 = ne.Sphere(3)
    s = ne.Sphere(3, element_sizes=(1, 1))

    # Should give exactly the same results since element_sizes are 1s
    for p in ((0, 0), (-23, 1)):
        assert_array_equal(s1(p), s(p))
        ok_(len(s(p)) == len(set(s(p))))

    # Raise exception if query dimensionality does not match element_sizes
    assert_raises(ValueError, s, (1,))

    s = ne.Sphere(3, element_sizes=(1.5, 2))
    assert_array_equal(s((0, 0)),
                       [(-2, 0), (-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 0), (0, 1),
                        (1, -1), (1, 0), (1, 1), (2, 0)])

    s = ne.Sphere(1.5, element_sizes=(1.5, 1.5, 1.5))
    res = s((0, 0, 0))
    ok_(np.all([np.sqrt(np.sum(np.array(x)**2)) <= 1.5 for x in res]))
    ok_(len(res) == 7)

    # all neighbors so no more than 1 voxel away -- just a cube, for
    # some "sphere" effect radius had to be 3.0 ;)
    td = np.sqrt(3*1.5**2)
    s = ne.Sphere(td, element_sizes=(1.5, 1.5, 1.5))
    res = s((0, 0, 0))
    ok_(np.all([np.sqrt(np.sum(np.array(x)**2)) <= td for x in res]))
    ok_(np.all([np.sum(np.abs(x) > 1) == 0 for x in res]))
    ok_(len(res) == 27)


def test_hollowsphere_basic():
    hs = ne.HollowSphere(1, 0)
    assert_array_equal(hs((2, 1)),  [(1, 1), (2, 0), (2, 2), (3, 1)])
    assert_array_equal(hs((1, )),   [(0,), (2,)])
    assert_equal(len(hs((1,1,1))), 6)


def test_hollowsphere_degenerate_neighborhood():
    """Test either we sustain empty neighborhoods
    """
    hs = ne.HollowSphere(1, inner_radius=0, element_sizes=(3,3,3))
    assert_equal(len(hs((1,1,1))), 0)


def test_hollowsphere_include_center():
    hs = ne.HollowSphere(1, 0, include_center=True)
    assert_array_equal(hs((2, 1)),  [(2,1), (1, 1), (2, 0), (2, 2), (3, 1)])
    assert_array_equal(hs((1, )),   [(1,), (0,), (2,)])
    assert_equal(len(hs((1,1,1))), 7)


def test_query_engine():
    data = np.arange(54)
    # indices in 3D
    ind = np.transpose((np.ones((3, 3, 3)).nonzero()))
    # sphere generator for 3 elements diameter
    sphere = ne.Sphere(1)
    # dataset with just one "space"
    ds = Dataset([data, data], fa={'s_ind': np.concatenate((ind, ind))})
    # and the query engine attaching the generator to the "index-space"
    qe = ne.IndexQueryEngine(s_ind=sphere)
    # cannot train since the engine does not know about the second space
    assert_raises(ValueError, qe.train, ds)
    # now do it again with a full spec
    ds = Dataset([data, data], fa={'s_ind': np.concatenate((ind, ind)),
                                   't_ind': np.repeat([0,1], 27)})
    qe = ne.IndexQueryEngine(s_ind=sphere, t_ind=None)
    qe.train(ds)
    # internal representation check
    # YOH: invalid for new implementation with lookup tables (dictionaries)
    #assert_array_equal(qe._searcharray,
    #                   np.arange(54).reshape(qe._searcharray.shape) + 1)
    # should give us one corner, collapsing the 't_ind'
    assert_array_equal(qe(s_ind=(0, 0, 0)),
                       [0, 1, 3, 9, 27, 28, 30, 36])
    # directly specifying an index for 't_ind' without having an ROI
    # generator, should give the same corner, but just once
    assert_array_equal(qe(s_ind=(0, 0, 0), t_ind=0), [0, 1, 3, 9])
    # just out of the mask -- no match
    assert_array_equal(qe(s_ind=(3, 3, 3)), [])
    # also out of the mask -- but single match
    assert_array_equal(qe(s_ind=(2, 2, 3), t_ind=1), [53])
    # query by id
    assert_array_equal(qe(s_ind=(0, 0, 0), t_ind=0), qe[0])
    assert_array_equal(qe(s_ind=(0, 0, 0), t_ind=[0, 1]),
                       qe(s_ind=(0, 0, 0)))
    # should not fail if t_ind is outside
    assert_array_equal(qe(s_ind=(0, 0, 0), t_ind=[0, 1, 10]),
                       qe(s_ind=(0, 0, 0)))

    # should fail if asked about some unknown thing
    assert_raises(ValueError, qe.__call__, s_ind=(0, 0, 0), buga=0)

    # Test by using some literal feature atttribute
    ds.fa['lit'] =  ['roi1', 'ro2', 'r3']*18
    # should work as well as before
    assert_array_equal(qe(s_ind=(0, 0, 0)), [0, 1, 3, 9, 27, 28, 30, 36])
    # should fail if asked about some unknown (yet) thing
    assert_raises(ValueError, qe.__call__, s_ind=(0,0,0), lit='roi1')

    # Create qe which can query literals as well
    qe_lit = ne.IndexQueryEngine(s_ind=sphere, t_ind=None, lit=None)
    qe_lit.train(ds)
    # should work as well as before
    assert_array_equal(qe_lit(s_ind=(0, 0, 0)), [0, 1, 3, 9, 27, 28, 30, 36])
    # and subselect nicely -- only /3 ones
    assert_array_equal(qe_lit(s_ind=(0, 0, 0), lit='roi1'),
                       [0, 3, 9, 27, 30, 36])
    assert_array_equal(qe_lit(s_ind=(0, 0, 0), lit=['roi1', 'ro2']),
                       [0, 1, 3, 9, 27, 28, 30, 36])


def test_cached_query_engine():
    """Test cached query engine
    """
    sphere = ne.Sphere(1)
    # dataset with just one "space"
    ds = datasets['3dlarge']
    qe0 = ne.IndexQueryEngine(myspace=sphere)
    qec = ne.CachedQueryEngine(qe0)

    # and ground truth one
    qe = ne.IndexQueryEngine(myspace=sphere)
    results_ind = []
    results_kw = []

    def cmp_res(res1, res2):
        comp = [x == y for x, y in zip(res1, res2)]
        ok_(np.all(comp))

    for iq, q in enumerate((qe, qec)):
        q.train(ds)
        # sequential train on the same should be ok in both cases
        q.train(ds)
        res_ind = [q[fid] for fid in xrange(ds.nfeatures)]
        res_kw = [q(myspace=x) for x in ds.fa.myspace]
        # test if results match
        cmp_res(res_ind, res_kw)

        results_ind.append(res_ind)
        results_kw.append(res_kw)

    # now check if results of cached were the same as of regular run
    cmp_res(results_ind[0], results_ind[1])

    # Now do sanity checks
    assert_raises(ValueError, qec.train, ds[:, :-1])
    assert_raises(ValueError, qec.train, ds.copy())
    ds2 = ds.copy()
    qec.untrain()
    qec.train(ds2)
    # should be the same results on the copy
    cmp_res(results_ind[0], [qec[fid] for fid in xrange(ds.nfeatures)])
    cmp_res(results_kw[0], [qec(myspace=x) for x in ds.fa.myspace])
    ok_(qec.train(ds2) is None)
    # unfortunately we are not catching those
    #ds2.fa.myspace = ds2.fa.myspace*3
    #assert_raises(ValueError, qec.train, ds2)

def test_scattered_neighborhoods():
    radius = 1
    sphere = ne.Sphere(radius)
    coords = range(50)

    scoords, sidx = ne.scatter_neighborhoods(sphere, coords,
                                             deterministic=False)
    # for this specific case of 1d coordinates the coords and idx should be
    # identical
    assert_array_equal(scoords, sidx)
    # minimal difference of successive coordinates must be larger than the
    # radius of the spheres. Test only works for 1d coords and sorted return
    # values
    assert(np.diff(scoords).min() > radius)

    # now the same for the case where a particular coordinate appears multiple
    # times
    coords = range(10) + range(10)
    scoords, sidx = ne.scatter_neighborhoods(sphere, coords,
                                             deterministic=False)
    sidx = sorted(sidx)
    assert_array_equal(scoords, sidx[:5])
    assert_array_equal(scoords, [i - 10 for i in sidx[5:]])
