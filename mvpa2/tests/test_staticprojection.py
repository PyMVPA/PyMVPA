# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA StaticProjectionMapper"""

import numpy as np
from mvpa2.testing import *
from mvpa2.testing.datasets import *
from mvpa2.mappers.staticprojection import StaticProjectionMapper

def test_staticprojection_reverse_fa():
    ds = datasets['uni2small']
    proj = np.eye(ds.nfeatures)
    spm = StaticProjectionMapper(proj=proj[:,:3], recon=proj[:,:3].T)

    ok_(len(ds.fa) > 0)                   # we have some fa
    dsf = spm.forward(ds)
    ok_(len(dsf.fa) == 0)                 # no fa were left
    assert_equal(dsf.nfeatures, 3)        # correct # of features
    assert_equal(dsf.fa.attr_length, 3)   # and .fa knows about that 
    dsf.fa['new3'] = np.arange(3)

    dsfr = spm.reverse(dsf)
    ok_(len(dsfr.fa) == 0)                 # no fa were left
    assert_equal(dsfr.nfeatures, 6)
    assert_equal(dsfr.fa.attr_length, 6)   # .fa knows about them again
    dsfr.fa['new'] = np.arange(6)

def test_staticprojection_pass_attr():
    ds = datasets['uni2small']
    proj = np.eye(ds.nfeatures)
    spm = StaticProjectionMapper(proj=proj[:,:3], recon=proj[:,:3].T, demean=False,
                enable_ca=['pass_fa'],pass_attr=[('ca.pass_fa','fa',1,'magic')])
    spm.ca.pass_fa = np.arange(3)
    ok_(len(ds.fa) > 0)
    dsf_fwd = spm.forward(ds)
    dsf = spm(ds)
    ok_(len(dsf.fa) > 0)
    assert_array_equal(dsf.samples, dsf_fwd.samples)
    assert_string_equal(dsf.fa.keys()[0], 'magic')
    assert_array_equal(dsf.fa.magic, np.arange(3))

