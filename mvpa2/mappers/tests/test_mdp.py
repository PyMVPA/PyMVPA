# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for basic mappers'''

import numpy as np

from mvpa2.testing import *
skip_if_no_external('mdp')

from mvpa2.base import externals
import mdp

from mvpa2.mappers.mdp_adaptor import MDPNodeMapper, MDPFlowMapper, PCAMapper, \
        ICAMapper
from mvpa2.mappers.lle import LLEMapper
from mvpa2.datasets.base import Dataset
from mvpa2.base.dataset import DAE
from mvpa2.misc.data_generators import normal_feature_dataset

@reseed_rng()
def test_mdpnodemapper():
    ds = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)

    node = mdp.nodes.PCANode()
    mm = MDPNodeMapper(node, nodeargs={'stoptrain': ((), {'debug': True})})

    mm.train(ds)

    fds = mm.forward(ds)
    if externals.versions['mdp'] >= '2.5':
        assert_true(hasattr(mm.node, 'cov_mtx'))

    assert_true(isinstance(fds, Dataset))
    assert_equal(fds.samples.shape, ds.samples.shape)

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
    dsbig = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=10)
    mm.train(dsbig)


def test_mdpflowmapper():
    flow = mdp.nodes.PCANode() + mdp.nodes.SFANode()
    fm = MDPFlowMapper(flow)
    ds = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)

    fm.train(ds)
    assert_false(fm.flow[0].is_training())
    assert_false(fm.flow[1].is_training())

    fds = fm.forward(ds)
    assert_true(isinstance(fds, Dataset))
    assert_equal(fds.samples.shape, ds.samples.shape)


def test_mdpflow_additional_arguments():
    skip_if_no_external('mdp', min_version='2.5')
    # we have no IdentityNode yet... is there analog?

    ds = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)
    flow = mdp.nodes.PCANode() + mdp.nodes.IdentityNode() + mdp.nodes.FDANode()
    # this is what it would look like in MDP itself
    #flow.train([[ds.samples],
    #            [[ds.samples, ds.sa.targets]]])
    assert_raises(ValueError, MDPFlowMapper, flow, node_arguments=[[],[]])
    fm = MDPFlowMapper(flow, node_arguments = ([], [], [DAE('sa', 'targets')]))
    fm.train(ds)
    fds = fm.forward(ds)
    assert_equal(ds.samples.shape, fds.samples.shape)
    rds = fm.reverse(fds)
    assert_array_almost_equal(ds.samples, rds.samples)

def test_mdpflow_additional_arguments_nones():
    skip_if_no_external('mdp', min_version='2.5')
    # we have no IdentityNode yet... is there analog?

    ds = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)
    flow = mdp.nodes.PCANode() + mdp.nodes.IdentityNode() + mdp.nodes.FDANode()
    # this is what it would look like in MDP itself
    #flow.train([[ds.samples],
    #            [[ds.samples, ds.sa.targets]]])
    assert_raises(ValueError, MDPFlowMapper, flow, node_arguments=[[],[]])
    fm = MDPFlowMapper(flow, node_arguments = (None, None, [ds.sa.targets]))
    fm.train(ds)
    fds = fm.forward(ds)
    assert_equal(ds.samples.shape, fds.samples.shape)
    rds = fm.reverse(fds)
    assert_array_almost_equal(ds.samples, rds.samples)

@reseed_rng()
def test_pcamapper():
    # data: 40 sample feature line in 20d space (40x20; samples x features)
    ndlin = Dataset(np.concatenate([np.arange(40)
                               for i in range(20)]).reshape(20,-1).T)

    pm = PCAMapper()
    # train PCA
    assert_raises(mdp.NodeException, pm.train, ndlin)
    ndlin.samples = ndlin.samples.astype('float')
    ndlin_noise = ndlin.copy()
    ndlin_noise.samples += np.random.random(size=ndlin.samples.shape)
    # we have no variance for more than one PCA component, hence just one
    # actual non-zero eigenvalue
    assert_raises(mdp.NodeException, pm.train, ndlin)
    pm.train(ndlin_noise)
    assert_equal(pm.proj.shape, (20, 20))
    # now project data into PCA space
    p = pm.forward(ndlin.samples)
    assert_equal(p.shape, (40, 20))
    # check that the mapped data can be fully recovered by 'reverse()'
    assert_array_almost_equal(pm.reverse(p), ndlin)


@reseed_rng()
def test_icamapper():
    # data: 40 sample feature line in 2d space (40x2; samples x features)
    samples = np.vstack([np.arange(40.) for i in range(2)]).T
    samples -= samples.mean()
    samples +=  np.random.normal(size=samples.shape, scale=0.1)
    ndlin = Dataset(samples)

    pm = ICAMapper()
    try:
        pm.train(ndlin.copy())
        assert_equal(pm.proj.shape, (2, 2))
        p = pm.forward(ndlin.copy())
        assert_equal(p.shape, (40, 2))
        # check that the mapped data can be fully recovered by 'reverse()'
        assert_array_almost_equal(pm.reverse(p), ndlin)
    except mdp.NodeException:
        # do not puke if the ICA did not converge at all -- that is not our
        # fault but MDP's
        pass


def test_llemapper():
    skip_if_no_external('mdp', min_version='2.4')

    ds = Dataset(np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                          [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                          [1., 1., 0.], [1., 1., 1.]]))
    pm = LLEMapper(3, output_dim=2)
    pm.train(ds)
    fmapped = pm.forward(ds)
    assert_equal(fmapped.shape, (8, 2))

@reseed_rng()
def test_nodeargs():
    skip_if_no_external('mdp', min_version='2.4')
    ds = normal_feature_dataset(perlabel=10, nlabels=2, nfeatures=4)

    for svd_val in [True, False]:
        pcm = PCAMapper(alg='PCA', svd=svd_val)
        assert_equal(pcm.node.svd, svd_val)
        pcm.train(ds)
        assert_equal(pcm.node.svd, svd_val)
    for output_dim in [0.5, 0.95, 0.99, 10, 50, 100]:
        pcm = PCAMapper(alg='PCA', output_dim=output_dim)
        for i in range(2):              # so we also test on trained one
            if isinstance(output_dim, float):
                assert_equal(pcm.node.desired_variance, output_dim)
            else:
                assert_equal(pcm.node.output_dim, output_dim)
            pcm.train(ds)
            if isinstance(output_dim, float):
                assert_not_equal(pcm.node.output_dim, output_dim)
                # some dimensions are chosen
                assert_true(pcm.node.output_dim > 0)

