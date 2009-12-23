# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os

import numpy as N
from numpy import array

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.datasets.base import Dataset
import mvpa.misc.neighborhood as ne
from mvpa import pymvpa_dataroot

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
    assert_array_equal(array(s(center1)), target)

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
    assert_raises(TypeError, ne.Sphere, 1, extent=(1.0,1.0,1.0))

    s = ne.Sphere(1)
    #assert_raises(ValueError, s, (1))
    # No float coordinates allowed for now... XXX might like to change that ;)
    assert_raises(ValueError, s, (1.0, 1.0, 1.0))

def test_query_engine():
    data = N.arange(54)
    # indices in 3D
    ind = N.transpose((N.ones((3,3,3)).nonzero()))
    # sphere generator for 3 elements diameter
    sphere = ne.Sphere(1)
    # dataset with just one "space"
    ds = Dataset([data,data], fa={'s_ind': N.concatenate((ind, ind))})
    # and the query engine attaching the generator to the "index-space"
    qe = ne.IndexQueryEngine(s_ind=sphere)
    # cannot train since the engine does not know about the second space
    assert_raises(ValueError, qe.train, ds)
    # now do it again with a full spec
    ds = Dataset([data, data], fa={'s_ind': N.concatenate((ind, ind)),
                                   't_ind': N.repeat([0,1], 27)})
    qe = ne.IndexQueryEngine(s_ind=sphere, t_ind=None)
    qe.train(ds)
    # internal representation check
    # YOH: invalid for new implementation with lookup tables (dictionaries)
    #assert_array_equal(qe._searcharray,
    #                   N.arange(54).reshape(qe._searcharray.shape) + 1)
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
    assert_raises(ValueError, qe.__call__, s_ind=(0,0,0), buga=0)

    # Test by using some literal feature atttribute
    ds.fa['lit'] =  ['roi1', 'ro2', 'r3']*18
    # should work as well as before
    assert_array_equal(qe(s_ind=(0,0,0)), [0, 1, 3, 9, 27, 28, 30, 36])
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

