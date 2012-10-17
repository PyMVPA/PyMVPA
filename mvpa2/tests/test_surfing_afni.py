# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA surface searchlight functions specific for 
handling AFNI datasets"""

import numpy as np
import nibabel as ni

import os
import tempfile

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.datasets import Dataset
from mvpa2.measures.base import Measure
from mvpa2.datasets.mri import fmri_dataset

from mvpa2.misc.surfing import volgeom, volsurf, sparse_attributes, \
                                        surf_voxel_selection

from mvpa2.support.nibabel import surf, surf_fs_asc, afni_niml, afni_niml_dset

class SurfTests(unittest.TestCase):
    """Test for AFNI I/O together with surface-based stuff

    NNO Aug 2012

    'Ground truth' is whatever output is returned by the implementation
    as of mid-Aug 2012"""

    def _get_rng(self):
        keys = [(17 * i ** 5 + 78234745 * i + 8934) % (2 ** 32 - 1)
                        for i in xrange(624)]
        keys = np.asanyarray(keys, dtype=np.uint32)
        rng = np.random.RandomState()
        rng.set_state(('MT19937', keys, 0))
        return rng

    def test_afni_niml_dset(self):
        sz = (100, 45) # dataset size
        rng = self._get_rng() # generate random data

        expected_vals = {(0, 0):-2.13856 , (sz[0] - 1, sz[1] - 1):-1.92434,
                         (sz[0], sz[1] - 1):None, (sz[0] - 1, sz[1]):None,
                         sz:None}

        # test for different formats in which the data is stored
        fmts = ['text', 'binary', 'base64']

        # also test for different datatypes
        tps = [np.int32, np.int64, np.float32, np.float64]

        # generated random data
        data = rng.normal(size=sz)

        # set labels for samples, and set node indices
        labels = ['lab_%d' % round(rng.uniform() * 1000)
                        for _ in xrange(sz[1])]
        node_indices = np.argsort(rng.uniform(size=(sz[0],)))
        node_indices = np.reshape(node_indices, (sz[0], 1))


        eps = .00001

        # test I/O
        _, fn = tempfile.mkstemp('data.niml.dset', 'test')

        # depending on the mode we do different tests (but on the same data)
        modes = ['normal', 'skipio', 'sparse2full']

        for fmt in fmts:
            for tp in tps:
                for mode in modes:
                    # make a dataset
                    dset = dict(data=np.asarray(data, tp),
                                labels=labels,
                                node_indices=node_indices)
                    dset_keys = dset.keys()

                    if mode == 'skipio':
                        # try conversion to/from raw NIML
                        # do not write to disk
                        r = afni_niml_dset.dset2rawniml(dset)
                        s = afni_niml.rawniml2string(r)
                        r2 = afni_niml.string2rawniml(s)
                        dset2 = afni_niml_dset.rawniml2dset(r2)[0]

                    else:
                        # write and read from disk
                        afni_niml_dset.write(fn, dset, fmt)
                        dset2 = afni_niml_dset.read(fn)
                        os.remove(fn)

                    # data in dset and dset2 should be identical
                    for k in dset_keys:
                        # general idea is to test whether v is equal to v2
                        v = dset[k]
                        v2 = dset2[k]

                        if k == 'data':
                            if mode == 'sparse2full':
                                # test the sparse2full feature
                                # this changes the order of the data over columns
                                # so we skip testing whether dset2 is equal to dset
                                nfull = 2 * sz[0]

                                dset3 = afni_niml_dset.sparse2full(dset2,
                                                            pad_to_node=nfull)

                                assert_equal(dset3['data'].shape[0], nfull)

                                idxs = dset['node_indices'][:, 0]
                                idxs3 = dset3['node_indices'][:, 0]
                                vbig = np.zeros((nfull, sz[1]))
                                vbig[idxs, :] = v[np.arange(sz[0]), :]
                                v = vbig
                                v2 = dset3['data'][idxs3, :]
                            else:
                                # check that data is as expected
                                for pos, val in expected_vals.iteritems():
                                    if val is None:
                                        assert_raises(IndexError, lambda x:x[pos], v2)
                                    else:
                                        val2 = np.asarray(val, tp)
                                        assert_true(abs(v2[pos] - val2) < eps)
                        if type(v) is list:
                            assert_equal(v, v2)
                        else:
                            eps_dec = 4
                            if mode != 'sparse2full' or k == 'data':
                                assert_array_almost_equal(v, v2, eps_dec)

def _test_afni_suma_spec():
    datapath = os.path.join(pymvpa_datadbroot,
                        'tutorial_data', 'tutorial_data', 'data', 'surfing')
    # TODO: test on surfing data


def suite():
    """Create the suite"""
    return unittest.makeSuite(SurfTests)


if __name__ == '__main__':
    import runner
