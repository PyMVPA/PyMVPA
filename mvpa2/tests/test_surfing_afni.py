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

import os
import tempfile

from mvpa2.testing import *

from mvpa2.support.nibabel import afni_niml, afni_niml_dset

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

    def test_afni_niml(self):
        # just a bunch of tests 

        ps = afni_niml._partial_string

        assert_equal(ps("", 0, 0), "")
        assert_equal(ps("ab", 0, 0), "")
        assert_equal(ps("abcdefghij", 0, 0), "")
        assert_equal(ps("", 2, 0), "")
        assert_equal(ps("ab", 2, 0), "")
        assert_equal(ps("abcdefghij", 2, 0), "")
        assert_equal(ps("", 0, 1), "")
        assert_equal(ps("ab", 0, 1), " ... b")
        assert_equal(ps("abcdefghij", 0, 1), " ... j")
        assert_equal(ps("", 2, 1), "")
        assert_equal(ps("ab", 2, 1), "")
        assert_equal(ps("abcdefghij", 2, 1), " ... j")
        assert_equal(ps("", 0, 100), "")
        assert_equal(ps("ab", 0, 100), "ab")
        assert_equal(ps("abcdefghij", 0, 100), "abcdefghij")
        assert_equal(ps("", 2, 100), "")
        assert_equal(ps("ab", 2, 100), "")
        assert_equal(ps("abcdefghij", 2, 100), "cdefghij")



        data = np.asarray([[1347506771, 1347506772],
                       [1347506773, 1347506774]],
                      dtype=np.int32)

        fmt_data_reprs = dict(text='1347506771 1347506772\n1347506773 1347506774',
                         binary='SRQPTRQPURQPVRQP',
                         base64='U1JRUFRSUVBVUlFQVlJRUA==')

        minimal_niml_struct = [{'dset_type': 'Node_Bucket',
                               'name': 'AFNI_dataset',
                               'ni_form': 'ni_group',
                               'nodes': [{'data': data,
                                          'data_type': 'Node_Bucket_data',
                                          'name': 'SPARSE_DATA',
                                          'ni_dimen': '2',
                                          'ni_type': '2*int32'},
                                         {'atr_name': 'COLMS_LABS',
                                          'data': 'col_0;col_1',
                                          'name': 'AFNI_atr',
                                          'ni_dimen': '1',
                                          'ni_type': 'String'}]}]


        def _eq(p, q):
            # helper function: equality for both arrays and other things
            return np.all(p == q) if type(p) is np.ndarray else p == q

        for fmt, data_repr in fmt_data_reprs.iteritems():
            s = afni_niml.rawniml2string(minimal_niml_struct, fmt)
            d = afni_niml.string2rawniml(s)

            # ensure data was converted properly
            assert_true(data_repr in s)

            for k, v in minimal_niml_struct[0].iteritems():
                if k == 'nodes':
                    # at least in one of the data
                    for node in v:
                        for kk, vv in node.iteritems():
                            # at least one of the data fields should have a value matching
                            # that from the expected converted value
                            dvals = [d[0]['nodes'][i].get(kk, None) for i in xrange(len(v))]
                            assert_true(any([_eq(vv, dval) for dval in dvals]))

                elif k != 'name':
                    # check header was properly converted
                    assert_true('%s="%s"' % (k, v) in s)


            # check that if we remove some important information, then parsing fails
            important_keys = ['ni_form', 'ni_dimen', 'ni_type']

            for k in important_keys:
                s_bad = s.replace(k, 'foo')
                assert_raises((KeyError, ValueError), afni_niml.string2rawniml, s_bad)

            # adding garbage at the beginning or end should fail the parse
            assert_raises((KeyError, ValueError), afni_niml.string2rawniml, s + "GARBAGE")
            assert_raises((KeyError, ValueError), afni_niml.string2rawniml, "GARBAGE" + s)



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
