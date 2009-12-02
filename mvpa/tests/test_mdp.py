
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for basic mappers'''

import numpy as N
import mdp

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.mappers.mdp import MDPNodeMapper, MDPFlowMapper
from mvpa.datasets.base import Dataset, DAE
from mvpa.misc.data_generators import normalFeatureDataset


def test_mdpnodemapper():
    ds = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=4)

    node = mdp.nodes.PCANode()
    mm = MDPNodeMapper(node, nodeargs={'stoptrain': ((), {'debug': True})})

    mm.train(ds)

    assert_equal(mm.get_insize(), 4)
    assert_true(N.all([mm.is_valid_inid(i) for i in range(4)]))
    assert_equal(mm.get_outsize(), 4)

    fds = mm.forward(ds)
    assert_true(hasattr(mm.node, 'cov_mtx'))

    assert_true(isinstance(fds, Dataset))
    assert_equal(fds.samples.shape, ds.samples.shape)
    assert_equal(mm.get_outsize(), ds.nfeatures)

    # set projection onto first 2 components
    mm.nodeargs['exec'] = ((), {'n': 2})
    #should be different from above
    lfds = mm.forward(ds.samples)
    # output shape changes although the node still claim otherwise
    assert_equal(mm.node.output_dim, 4)
    assert_equal(lfds.shape[0], fds.samples.shape[0])
    assert_equal(lfds.shape[1], 2)
    assert_array_equal(lfds, fds.samples[:, :2])

    # reverse
    rfds = mm.reverse(fds)

    # even smaller size works
    rlfds = mm.reverse(lfds)
    assert_equal(rfds.samples.shape, ds.samples.shape)

    # retraining has to work on a new dataset too, since we copy the node
    # internally
    dsbig = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=10)
    mm.train(dsbig)
    assert_equal(mm.get_outsize(), 10)


def test_mdpflowmapper():
    flow = mdp.nodes.PCANode() + mdp.nodes.SFANode()
    fm = MDPFlowMapper(flow)
    ds = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=4)

    fm.train(ds)
    assert_equal(fm.get_insize(), 4)
    assert_true(N.all([fm.is_valid_inid(i) for i in range(4)]))
    assert_equal(fm.get_outsize(), 4)
    assert_false(fm.flow[0].is_training())
    assert_false(fm.flow[1].is_training())

    fds = fm.forward(ds)
    assert_true(isinstance(fds, Dataset))
    assert_equal(fds.samples.shape, ds.samples.shape)
    assert_equal(fm.get_outsize(), ds.nfeatures)


def test_mdpmadness():
    ds = normalFeatureDataset(perlabel=10, nlabels=2, nfeatures=4)
    flow = mdp.nodes.PCANode() + mdp.nodes.FDANode()
    #flow.train([[ds.samples],
    #            [[ds.samples, ds.sa.labels]]])

    fm = MDPFlowMapper(flow,
                       data_iterables = [[],
                                         [DAE('sa', 'labels')]])
    fm.train(ds)
