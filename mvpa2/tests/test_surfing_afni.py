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

from mvpa2.support.nibabel import afni_niml, afni_niml_dset, afni_niml_roi, \
                                                surf, afni_suma_spec
from mvpa2.datasets import niml
from mvpa2.datasets.base import Dataset


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
                    assert_true(('%s="%s"' % (k, v)).encode() in s)


            # check that if we remove some important information, then parsing fails
            important_keys = ['ni_form', 'ni_dimen', 'ni_type']

            for k in important_keys:
                s_bad = s.replace(k.encode(), b'foo')
                assert_raises((KeyError, ValueError), afni_niml.string2rawniml, s_bad)

            # adding garbage at the beginning or end should fail the parse
            garbage = "GARBAGE".encode()
            assert_raises((KeyError, ValueError), afni_niml.string2rawniml, s + garbage)
            assert_raises((KeyError, ValueError), afni_niml.string2rawniml, garbage + s)


    @with_tempfile('.niml.dset', 'dset')
    def test_afni_niml_dset_with_2d_strings(self, fn):
        # test for 2D arrays with strings. These are possibly SUMA-incompatible
        # but should still be handled properly for i/o.
        # Addresses https://github.com/PyMVPA/PyMVPA/issues/163 (#163)
        samples = np.asarray([[1, 2, 3], [4, 5, 6]])
        labels = np.asarray(map(list, ['abcd', 'efgh']))
        idxs = np.asarray([np.arange(10, 14), np.arange(20, 24)])

        ds = Dataset(samples, sa=dict(labels=labels, idxs=idxs))

        for fmt in ('binary', 'text', 'base64'):
            niml.write(fn, ds, fmt)

            ds_ = niml.read(fn)

            assert_array_equal(ds.samples, ds_.samples)

            for sa_key in ds.sa.keys():
                v = ds.sa[sa_key].value
                v_ = ds_.sa[sa_key].value
                assert_array_equal(v, v_)


    @with_tempfile('.niml.dset', 'dset')
    def test_afni_niml_dset(self, fn):
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

    @with_tempfile('.niml.dset', 'dset')
    def test_niml(self, fn):
        d = dict(data=np.random.normal(size=(10, 2)),
              node_indices=np.arange(10),
              stats=['none', 'Tstat(2)'],
              labels=['foo', 'bar'])
        a = niml.from_niml(d)
        b = niml.to_niml(a)

        afni_niml_dset.write(fn, b)
        bb = afni_niml_dset.read(fn)
        cc = niml.from_niml(bb)

        os.remove(fn)

        for dset in (a, cc):
            assert_equal(list(dset.sa['labels']), d['labels'])
            assert_equal(list(dset.sa['stats']), d['stats'])
            assert_array_equal(np.asarray(dset.fa['node_indices']).ravel(),
                               d['node_indices'])

            eps_dec = 4
            assert_array_almost_equal(dset.samples, d['data'].transpose(),
                                                                    eps_dec)

        # some more tests to ensure that the order of elements is ok
        # (row first or column first)

        d = np.arange(10).reshape((5, -1)) + .5
        ds = Dataset(d)

        writers = [niml.write, afni_niml_dset.write]
        for i, writer in enumerate(writers):
            for form in ('text', 'binary', 'base64'):
                if i == 0:
                    writer(fn, ds, form=form)
                else:
                    writer(fn, dict(data=d.transpose()), form=form)

                x = afni_niml_dset.read(fn)
                assert_array_equal(x['data'], d.transpose())


    @with_tempfile('.niml.dset', 'dset')
    def test_niml_dset_voxsel(self, fn):
        if not externals.exists('nibabel'):
            return

        # This is actually a bit of an integration test.
        # It tests storing and retrieving searchlight results.
        # Imports are inline here so that it does not mess up the header
        # and makes the other unit tests more modular
        # XXX put this in a separate file?
        from mvpa2.misc.surfing import volgeom, surf_voxel_selection, queryengine
        from mvpa2.measures.searchlight import Searchlight
        from mvpa2.support.nibabel import surf
        from mvpa2.measures.base import Measure
        from mvpa2.datasets.mri import fmri_dataset

        class _Voxel_Count_Measure(Measure):
            # used to check voxel selection results
            is_trained = True
            def __init__(self, dtype, **kwargs):
                Measure.__init__(self, **kwargs)
                self.dtype = dtype

            def _call(self, dset):
                return self.dtype(dset.nfeatures)

        sh = (20, 20, 20)
        vg = volgeom.VolGeom(sh, np.identity(4))

        density = 20

        outer = surf.generate_sphere(density) * 10. + 5
        inner = surf.generate_sphere(density) * 5. + 5

        intermediate = outer * .5 + inner * .5
        xyz = intermediate.vertices

        radius = 50

        sel = surf_voxel_selection.run_voxel_selection(radius, vg, inner, outer)
        qe = queryengine.SurfaceVerticesQueryEngine(sel)

        for dtype in (int, float):
            sl = Searchlight(_Voxel_Count_Measure(dtype), queryengine=qe)

            ds = fmri_dataset(vg.get_empty_nifti_image(1))
            r = sl(ds)

            niml.write(fn, r)
            rr = niml.read(fn)

            os.remove(fn)

            assert_array_equal(r.samples, rr.samples)


    def test_niml_dset_stack(self):
        values = map(lambda x:np.random.normal(size=x), [(10, 3), (10, 4), (10, 5)])
        indices = [[0, 1, 2], [3, 2, 1, 0], None]

        dsets = []
        for v, i in zip(values, indices):
            dset = Dataset(v)
            if not i is None:
                dset.fa['node_indices'] = i
            dsets.append(dset)


        dset = niml.hstack(dsets)
        assert_equal(dset.nfeatures, 12)
        assert_equal(dset.nsamples, 10)
        indices = np.asarray([ 0, 1, 2, 6, 5, 4, 3, 7, 8, 9, 10, 11])
        assert_array_equal(dset.fa['node_indices'], indices)

        dset = niml.hstack(dsets, 10)
        dset = niml.hstack(dsets, 10) # twice to ensure not overwriting
        assert_equal(dset.nfeatures, 30)
        indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                              13, 12, 11, 10, 14, 15, 16, 17, 18, 19,
                              20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
        assert_array_equal(dset.fa['node_indices'], indices)

        assert_true(np.all(dset[:, 4].samples == 0))
        assert_array_equal(dset[:, 10:14].samples, dsets[1].samples)

        # If not enough space it should raise an error
        stacker = (lambda x: niml.hstack(dsets, x))
        assert_raises(ValueError, stacker, 2)

        # If sparse then with no padding it should fail
        dsets[0].fa.node_indices[0] = 3
        assert_raises(ValueError, stacker, None)

        # Using an illegal node index should raise an error
        dsets[1].fa.node_indices[0] = 666
        assert_raises(ValueError, stacker, 10)

    @with_tempfile('.niml.roi', 'dset')
    def test_afni_niml_roi(self, fn):
        payload = """# <Node_ROI
#  ni_type = "SUMA_NIML_ROI_DATUM"
#  ni_dimen = "5"
#  self_idcode = "XYZ_QlRYtdSyHmNr39qZWxD0wQ"
#  domain_parent_idcode = "XYZ_V_Ug6er2LCNoLy_OzxPsZg"
#  Parent_side = "no_side"
#  Label = "myroi"
#  iLabel = "12"
#  Type = "2"
#  ColPlaneName = "ROI.-.CoMminfl"
#  FillColor = "0.525490 0.043137 0.231373 1.000000"
#  EdgeColor = "0.000000 0.000000 1.000000 1.000000"
#  EdgeThickness = "2"
# >
 1 4 1 42946
 1 4 10 42946 42947 43062 43176 43289 43401 43512 43513 43623 43732
 1 4 8 43732 43623 43514 43404 43293 43181 43068 42954
 3 4 9 42954 42953 42952 42951 42950 42949 42948 42947 42946
 4 1 14 43063 43064 43065 43066 43067 43177 43178 43179 43180 43290 43291 43292 43402 43403
# </Node_ROI>"""

        with open(fn, 'w') as f:
            f.write(payload)

        rois = afni_niml_roi.read(fn)

        assert_equal(len(rois), 1)
        roi = rois[0]

        expected_keys = ['ni_type', 'ColPlaneName', 'iLabel', 'Parent_side',
                       'EdgeColor', 'Label', 'edges', 'ni_dimen',
                       'self_idcode', 'EdgeThickness', 'Type', 'areas',
                       'domain_parent_idcode', 'FillColor']

        assert_equal(set(roi.keys()), set(expected_keys))

        assert_equal(roi['Label'], 'myroi')
        assert_equal(roi['iLabel'], 12)

        # check edges
        arr = np.asarray
        expected_edges = [arr([42946]),
                          arr([42946, 42947, 43062, 43176, 43289, 43401,
                               43512, 43513, 43623, 43732]),
                          arr([43732, 43623, 43514, 43404, 43293, 43181,
                               43068, 42954]),
                          arr([42954, 42953, 42952, 42951, 42950, 42949,
                               42948, 42947, 42946])]

        for i in xrange(4):
            assert_array_equal(roi['edges'][i], expected_edges[i])

        # check nodes
        expected_nodes = [arr([43063, 43064, 43065, 43066, 43067, 43177, 43178,
                            43179, 43180, 43290, 43291, 43292, 43402, 43403])]

        assert_equal(len(roi['areas']), 1)
        assert_array_equal(roi['areas'][0], expected_nodes[0])


        # check mapping
        m = afni_niml_roi.read_mapping(rois)
        assert_equal(m.keys(), ['myroi'])

        unique_nodes = np.unique(expected_nodes[0])
        assert_array_equal(m['myroi'], unique_nodes)


    @with_tempfile()
    def test_afni_suma_spec(self, temp_dir):

        # XXX this function generates quite a few temporary files,
        #     which are removed at the end.
        #     the decorator @with_tempfile seems unsuitable as it only
        #     supports a single temporary file

        # make temporary directory
        os.mkdir(temp_dir)

        # generate surfaces
        inflated_surf = surf.generate_plane((0, 0, 0), (0, 1, 0), (0, 0, 1),
                                                    10, 10)
        white_surf = inflated_surf + 1.

        # helper function
        _tmp = lambda x:os.path.join(temp_dir, x)


        # filenames for surfaces and spec file
        inflated_fn = _tmp('_lh_inflated.asc')
        white_fn = _tmp('_lh_white.asc')
        spec_fn = _tmp('lh.spec')

        spec_dir = os.path.split(spec_fn)[0]

        # generate SUMA-like spec dictionary
        white = dict(SurfaceFormat='ASCII',
            EmbedDimension='3',
            SurfaceType='FreeSurfer',
            SurfaceName=white_fn,
            Anatomical='Y',
            LocalCurvatureParent='SAME',
            LocalDomainParent='SAME',
            SurfaceState='smoothwm')

        inflated = dict(SurfaceFormat='ASCII',
            EmbedDimension='3',
            SurfaceType='FreeSurfer',
            SurfaceName=inflated_fn,
            Anatomical='N',
            LocalCurvatureParent=white_fn,
            LocalDomainParent=white_fn,
            SurfaceState='inflated')

        # make SurfaceSpec object
        spec = afni_suma_spec.SurfaceSpec([white], directory=spec_dir)
        spec.add_surface(inflated)

        # test __str__ and __repr__
        assert_true('SurfaceSpec instance with 2 surfaces'
                        ', 2 states ' in '%s' % spec)
        assert_true(('%r' % spec).startswith('SurfaceSpec'))

        # test finding surfaces
        inflated_ = spec.find_surface_from_state('inflated')
        assert_equal([(1, inflated)], inflated_)

        empty = spec.find_surface_from_state('unknown')
        assert_equal(empty, [])

        # test .same_states
        minimal = afni_suma_spec.SurfaceSpec([dict(SurfaceState=s)
                                            for s in ('smoothwm', 'inflated')])
        assert_true(spec.same_states(minimal))
        assert_false(spec.same_states(afni_suma_spec.SurfaceSpec(dict())))

        # test 'smart' surface file matching
        assert_equal(spec.get_surface_file('smo'), white_fn)
        assert_equal(spec.get_surface_file('inflated'), inflated_fn)
        assert_equal(spec.get_surface_file('this should be None'), None)

        # test i/o
        spec.write(spec_fn)
        spec_ = afni_suma_spec.from_any(spec_fn)

        # prepare for another (right-hemisphere) spec file
        lh_spec = spec
        rh_spec_fn = spec_fn.replace('lh', 'rh')

        rh_inflated_fn = _tmp(os.path.split(inflated_fn)[1].replace('_lh',
                                                                    '_rh'))
        rh_white_fn = _tmp(os.path.split(white_fn)[1].replace('_lh',
                                                              '_rh'))
        rh_spec_fn = _tmp('rh.spec')

        rh_white = dict(SurfaceFormat='ASCII',
            EmbedDimension='3',
            SurfaceType='FreeSurfer',
            SurfaceName=rh_white_fn,
            Anatomical='Y',
            LocalCurvatureParent='SAME',
            LocalDomainParent='SAME',
            SurfaceState='smoothwm')

        rh_inflated = dict(SurfaceFormat='ASCII',
            EmbedDimension='3',
            SurfaceType='FreeSurfer',
            SurfaceName=rh_inflated_fn,
            Anatomical='N',
            LocalCurvatureParent=rh_white_fn,
            LocalDomainParent=rh_white_fn,
            SurfaceState='inflated')

        rh_spec = afni_suma_spec.SurfaceSpec([rh_white], directory=spec_dir)
        rh_spec.add_surface(rh_inflated)

        # write files
        all_temp_fns = [spec_fn, rh_spec_fn]
        for fn, s in [(rh_inflated_fn, inflated_surf),
                      (rh_white_fn, white_surf),
                      (inflated_fn, inflated_surf),
                      (white_fn, white_surf)]:
            surf.write(fn, s)
            all_temp_fns.append(fn)

        # test adding views
        added_specs = afni_suma_spec.hemi_pairs_add_views((lh_spec, rh_spec),
                                                          'inflated', '.asc')

        for hemi, added_spec in zip(('l', 'r'), added_specs):
            states = ['smoothwm', 'inflated'] + ['CoM%sinflated' % i
                                                    for i in 'msiap']
            assert_equal(states, [s['SurfaceState']
                                  for s in added_specs[0].surfaces])
            all_temp_fns.extend([s['SurfaceName']
                                 for s in added_spec.surfaces])

        # test combining specs (bh=both hemispheres)
        bh_spec = afni_suma_spec.combine_left_right(added_specs)

        # test merging specs (mh=merged hemispheres)
        mh_spec, mh_surfs = afni_suma_spec.merge_left_right(bh_spec)

        assert_equal([s['SurfaceState'] for s in mh_spec.surfaces],
                    ['smoothwm'] + ['CoM%sinflated' % i for i in 'msiap'])




def suite():  # pragma: no cover
    """Create the suite"""
    return unittest.makeSuite(SurfTests)

if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
