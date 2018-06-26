# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-A
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for Connectivity Hyperalignment."""

import unittest
import numpy as np

from mvpa2.misc.fx import get_random_rotation
from mvpa2.mappers.zscore import zscore
from mvpa2.datasets.base import Dataset
from mvpa2.support.nibabel.surf import Surface
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine

from mvpa2.algorithms.connectivity_hyperalignment import ConnectivityHyperalignment
from mvpa2.testing import skip_if_no_external


class ConnectivityHyperalignmentTests(unittest.TestCase):
    def get_testdata(self):
        # rs = np.random.RandomState(0)
        rs = np.random.RandomState()
        nt = 200
        n_triangles = 4
        ns = 10
        nv = n_triangles * 3
        vertices = np.zeros((nv, 3))  # 4 separated triangles
        faces = []
        for i in range(n_triangles):
            vertices[i*3] = [i*2, 0, 0]
            vertices[i*3+1] = [i*2+1, 1/np.sqrt(3), 0]
            vertices[i*3+2] = [i*2+1, -1/np.sqrt(3), 0]
            faces.append([i*3, i*3+1, i*3+2])
        faces = np.array(faces)
        surface = Surface(vertices, faces)

        ds_orig = np.zeros((nt, nv))
        # add coarse-scale information
        for i in range(n_triangles):
            ds_orig[:, i*3:(i+1)*3] += rs.normal(size=(nt, 1))
        # add fine-scale information
        ds_orig += rs.normal(size=(nt, nv))
        dss_train, dss_test = [], []
        for i in range(ns):
            ds = np.zeros_like(ds_orig)
            for j in range(n_triangles):
                ds[:, j*3:(j+1)*3] = np.dot(ds_orig[:, j*3:(j+1)*3],
                                            get_random_rotation(3))
                                            # special_ortho_group.rvs(3, random_state=rs))
            ds = Dataset(ds)
            ds.fa['node_indices'] = np.arange(nv)
            ds_train, ds_test = ds[:nt//2, :], ds[nt//2:, :]
            zscore(ds_train, chunks_attr=None)
            zscore(ds_test, chunks_attr=None)
            dss_train.append(ds_train)
            dss_test.append(ds_test)
        return dss_train, dss_test, surface

    def compute_connectivity_profile_similarity(self, dss):
        # from scipy.spatial.distance import pdist, squareform
        # conns = [1 - squareform(pdist(ds.samples.T, 'correlation')) for ds in dss]
        conns = [np.corrcoef(ds.samples.T) for ds in dss]
        conn_sum = np.sum(conns, axis=0)
        sim = np.zeros((len(dss), dss[0].shape[1]))
        for i, conn in enumerate(conns):
            conn_diff = conn_sum - conn
            zscore(conn_diff, chunks_attr=None)
            zscore(conn, chunks_attr=None)
            sim[i] = np.mean(conn_diff * conn, axis=0)
        return sim

    def test_connectivity_hyperalignment(self):
        skip_if_no_external('scipy')
        skip_if_no_external('hdf5')  # needed for default results backend hdf5

        dss_train, dss_test, surface = self.get_testdata()
        qe = SurfaceQueryEngine(surface, 10, fa_node_key='node_indices')
        cha = ConnectivityHyperalignment(
            mask_ids=[0, 3, 6, 9],
            seed_indices=[0, 3, 6, 9],
            seed_queryengines=qe,
            queryengine=qe)
        mappers = cha(dss_train)
        aligned_train = [mapper.forward(ds) for ds, mapper in zip(dss_train, mappers)]
        aligned_test = [mapper.forward(ds) for ds, mapper in zip(dss_test, mappers)]
        for ds in aligned_train + aligned_test:
            zscore(ds, chunks_attr=None)
        sim_train_before = self.compute_connectivity_profile_similarity(dss_train)
        sim_train_after = self.compute_connectivity_profile_similarity(aligned_train)
        sim_test_before = self.compute_connectivity_profile_similarity(dss_test)
        sim_test_after = self.compute_connectivity_profile_similarity(aligned_test)
        # ISC should be higher after CHA for both training and testing data
        self.assertTrue(sim_train_after.mean() > sim_train_before.mean())
        self.assertTrue(sim_test_after.mean() > sim_test_before.mean())
