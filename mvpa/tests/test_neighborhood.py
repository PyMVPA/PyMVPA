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

from mvpa.datasets import nifti
import mvpa.misc.neighborhood as ne
from mvpa import pymvpa_dataroot

def test_sphere():
    s = ne.Sphere(3)
    assert_equal(len(s.coord_list), 7)
    target = array([array([-1,  0,  0]),
              array([ 0, -1,  0]),
              array([ 0,  0, -1]),
              array([0, 0, 0]),
              array([0, 0, 1]),
              array([0, 1, 0]),
              array([1, 0, 0])])
    assert_array_equal(s.coord_list, target)

    target = [array([0, 1, 1]),
              array([1, 0, 1]),
              array([1, 1, 0]),
              array([1, 1, 1]),
              array([1, 1, 2]),
              array([1, 2, 1]),
              array([2, 1, 1])]
    assert_array_equal(array(s((1,1,1))), target)

    s = ne.Sphere(9)
    assert_equal(len(s.coord_list), 257)

def test_query_enigne():
    # load example 4 d dataset
    data = nifti.nifti_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
            labels=[1,2], space='voxel')
    s = ne.Sphere(9)
    qe = ne.QueryEngine(voxel=s)
    qe.train(data)
    #print "length:", len(qe(100000)[0])



