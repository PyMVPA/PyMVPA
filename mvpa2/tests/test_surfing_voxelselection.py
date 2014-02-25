# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA surface searchlight voxel selection"""

from mvpa2.testing import *
skip_if_no_external('nibabel')

import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal, \
    assert_raises

import nibabel as nb

import os
import tempfile

from mvpa2.testing import  reseed_rng
from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.datasets import Dataset
from mvpa2.measures.base import Measure
from mvpa2.datasets.mri import fmri_dataset

from mvpa2.support.nibabel import surf
from mvpa2.misc.surfing import surf_voxel_selection, queryengine, volgeom, \
                                volsurf
from mvpa2.misc.surfing.volume_mask_dict import VolumeMaskDictionary
from mvpa2.misc.surfing import volume_mask_dict

from mvpa2.measures.searchlight import Searchlight
from mvpa2.misc.surfing.queryengine import SurfaceVerticesQueryEngine, \
                                           SurfaceVoxelsQueryEngine, \
                                           disc_surface_queryengine

from mvpa2.measures.base import Measure, \
        TransferMeasure, RepeatedMeasure, CrossValidation
from mvpa2.clfs.smlr import SMLR
from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.mappers.fx import mean_sample
from mvpa2.misc.io.base import SampleAttributes
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.neighborhood import Sphere, IndexQueryEngine
from mvpa2.clfs.gnb import GNB

if externals.exists('h5py'):
    from mvpa2.base.hdf5 import h5save, h5load


class SurfVoxelSelectionTests(unittest.TestCase):

    def test_voxel_selection(self):
        '''Compare surface and volume based searchlight'''

        '''
        Tests to see whether results are identical for surface-based
        searchlight (just one plane; Euclidean distnace) and volume-based
        searchlight.

        Note that the current value is a float; if it were int, it would
        specify the number of voxels in each searchlight'''

        radius = 10.

        '''Define input filenames'''
        epi_fn = os.path.join(pymvpa_dataroot, 'bold.nii.gz')
        maskfn = os.path.join(pymvpa_dataroot, 'mask.nii.gz')

        '''
        Use the EPI datafile to define a surface.
        The surface has as many nodes as there are voxels
        and is parallel to the volume 'slice'
        '''
        vg = volgeom.from_any(maskfn, mask_volume=True)

        aff = vg.affine
        nx, ny, nz = vg.shape[:3]

        '''Plane goes in x and y direction, so we take these vectors
        from the affine transformation matrix of the volume'''
        plane = surf.generate_plane(aff[:3, 3], aff[:3, 0], aff[:3, 1],
                                    nx, ny)



        '''
        Simulate pial and white matter as just above and below
        the central plane
        '''
        normal_vec = aff[:3, 2]
        outer = plane + normal_vec
        inner = plane + -normal_vec

        '''
        Combine volume and surface information
        '''
        vsm = volsurf.VolSurfMaximalMapping(vg, outer, inner)

        '''
        Run voxel selection with specified radius (in mm), using
        Euclidean distance measure
        '''
        surf_voxsel = surf_voxel_selection.voxel_selection(vsm, radius,
                                                    distance_metric='e')

        '''Define the measure'''

        # run_slow=True would give an actual cross-validation with meaningful
        # accuracies. Because this is a unit-test only the number of voxels
        # in each searchlight is tested.
        run_slow = False

        if run_slow:
            meas = CrossValidation(GNB(), OddEvenPartitioner(),
                                   errorfx=lambda p, t: np.mean(p == t))
            postproc = mean_sample
        else:
            meas = _Voxel_Count_Measure()
            postproc = lambda x:x

        '''
        Surface analysis: define the query engine, cross validation,
        and searchlight
        '''
        surf_qe = SurfaceVerticesQueryEngine(surf_voxsel)
        surf_sl = Searchlight(meas, queryengine=surf_qe, postproc=postproc)


        '''
        new (Sep 2012): also test 'simple' queryengine wrapper function
        '''

        surf_qe2 = disc_surface_queryengine(radius, maskfn, inner, outer,
                                            plane, volume_mask=True,
                                            distance_metric='euclidean')
        surf_sl2 = Searchlight(meas, queryengine=surf_qe2,
                               postproc=postproc)


        '''
        Same for the volume analysis
        '''
        element_sizes = tuple(map(abs, (aff[0, 0], aff[1, 1], aff[2, 2])))
        sph = Sphere(radius, element_sizes=element_sizes)
        kwa = {'voxel_indices': sph}

        vol_qe = IndexQueryEngine(**kwa)
        vol_sl = Searchlight(meas, queryengine=vol_qe, postproc=postproc)


        '''The following steps are similar to start_easy.py'''
        attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                                'attributes_literal.txt'))

        mask = surf_voxsel.get_mask()

        dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot,
                                                    'bold.nii.gz'),
                               targets=attr.targets,
                               chunks=attr.chunks,
                               mask=mask)

        if run_slow:
            # do chunkswise linear detrending on dataset

            poly_detrend(dataset, polyord=1, chunks_attr='chunks')

            # zscore dataset relative to baseline ('rest') mean
            zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']))

        # select class face and house for this demo analysis
        # would work with full datasets (just a little slower)
        dataset = dataset[np.array([l in ['face', 'house']
                                    for l in dataset.sa.targets],
                                    dtype='bool')]

        '''Apply searchlight to datasets'''
        surf_dset = surf_sl(dataset)
        surf_dset2 = surf_sl2(dataset)
        vol_dset = vol_sl(dataset)

        surf_data = surf_dset.samples
        surf_data2 = surf_dset2.samples
        vol_data = vol_dset.samples

        assert_array_equal(surf_data, surf_data2)
        assert_array_equal(surf_data, vol_data)

    def test_voxel_selection_alternative_calls(self):
        # Tests a multitude of different searchlight calls
        # that all should yield exactly the same results.
        #
        # Calls differ by whether the arguments are filenames
        # or data objects, whether values are specified explicityly
        # or set to the default implicitly (using None).
        # and by different calls to run the voxel selection.
        #
        # This method does not test for mask functionality.

        # define the volume
        vol_shape = (10, 10, 10, 3)
        vol_affine = np.identity(4)
        vol_affine[0, 0] = vol_affine[1, 1] = vol_affine[2, 2] = 5



        # four versions: array, nifti image, file name, fmri dataset
        volarr = np.ones(vol_shape)
        volimg = nb.Nifti1Image(volarr, vol_affine)
        # There is a detected problem with elderly NumPy's (e.g. 1.6.1
        # on precise on travis) leading to segfaults while operating
        # on memmapped volumes being forwarded to pprocess.
        # Thus just making it compressed volume for those cases
        suf = '.gz' \
            if externals.exists('pprocess') and externals.versions['numpy'] < '1.6.2' \
            else ''
        fd, volfn = tempfile.mkstemp('vol.nii' + suf, 'test'); os.close(fd)
        volimg.to_filename(volfn)
        volds = fmri_dataset(volfn)

        fd, volfngz = tempfile.mkstemp('vol.nii.gz', 'test'); os.close(fd)
        volimg.to_filename(volfngz)
        voldsgz = fmri_dataset(volfngz)


        # make the surfaces
        sphere_density = 10

        # two versions: Surface and file name
        outer = surf.generate_sphere(sphere_density) * 25. + 15
        inner = surf.generate_sphere(sphere_density) * 20. + 15
        intermediate = inner * .5 + outer * .5
        nv = outer.nvertices

        fd, outerfn = tempfile.mkstemp('outer.asc', 'test'); os.close(fd)
        fd, innerfn = tempfile.mkstemp('inner.asc', 'test'); os.close(fd)
        fd, intermediatefn = tempfile.mkstemp('intermediate.asc', 'test'); os.close(fd)

        for s, fn in zip([outer, inner, intermediate],
                         [outerfn, innerfn, intermediatefn]):
            surf.write(fn, s, overwrite=True)

        # searchlight radius (in mm)
        radius = 10.

        # dataset used to run searchlight on
        ds = fmri_dataset(volfn)

        # simple voxel counter (run for each searchlight position)
        m = _Voxel_Count_Measure()

        # number of voxels expected in each searchlight
        r_expected = np.array([[18, 9, 10, 9, 9, 9, 9, 10, 9,
                                 9, 9, 9, 11, 11, 11, 11, 10,
                                10, 10, 9, 10, 11, 9, 10, 10,
                                8, 7, 8, 8, 8, 9, 10, 12, 12,
                                11, 7, 7, 8, 5, 9, 11, 11, 12,
                                12, 9, 5, 8, 7, 7, 12, 12, 13,
                                12, 12, 7, 7, 8, 5, 9, 12, 12,
                                13, 11, 9, 5, 8, 7, 7, 11, 12,
                                12, 11, 12, 10, 10, 11, 9, 11,
                                12, 12, 12, 12, 16, 13, 16, 16,
                                16, 17, 15, 17, 17, 17, 16, 16,
                                16, 18, 16, 16, 16, 16, 18, 16]])



        params = dict(intermediate_=(intermediate, intermediatefn, None),
                      center_nodes_=(None, range(nv)),
                      volume_=(volimg, volfn, volds, volfngz, voldsgz),
                      surf_src_=('filename', 'surf'),
                      volume_mask_=(None, True, 0, 2),
                      call_method_=("qe", "rvs", "gam"))

        combis = _cartprod(params) # compute all possible combinations
        combistep = 17  #173
                        # some fine prime number to speed things up
                        # if this value becomes too big then not all
                        # cases are covered
                        # the unit test tests itself whether all values
                        # occur at least once

        tested_params = dict()
        def val2str(x):
            return '%r:%r' % (type(x), x)

        for i in xrange(0, len(combis), combistep):
            combi = combis[i]

            intermediate_ = combi['intermediate_']
            center_nodes_ = combi['center_nodes_']
            volume_ = combi['volume_']
            surf_src_ = combi['surf_src_']
            volume_mask_ = combi['volume_mask_']
            call_method_ = combi['call_method_']


            # keep track of which values were used -
            # so that this unit test tests itself

            for k in combi.keys():
                if not k in tested_params:
                    tested_params[k] = set()
                tested_params[k].add(val2str(combi[k]))

            if surf_src_ == 'filename':
                s_i, s_m, s_o = inner, intermediate, outer
            elif surf_src_ == 'surf':
                s_i, s_m, s_o = innerfn, intermediatefn, outerfn
            else:
                raise ValueError('this should not happen')

            if call_method_ == "qe":
                # use the fancy query engine wrapper
                qe = disc_surface_queryengine(radius,
                        volume_, s_i, s_o, s_m,
                        source_surf_nodes=center_nodes_,
                        volume_mask=volume_mask_)
                sl = Searchlight(m, queryengine=qe)
                r = sl(ds).samples

            elif call_method_ == 'rvs':
                # use query-engine but build the
                # ingredients by hand
                vg = volgeom.from_any(volume_,
                                      volume_mask_)
                vs = volsurf.VolSurfMaximalMapping(vg, s_i, s_o)
                sel = surf_voxel_selection.voxel_selection(
                        vs, radius, source_surf=s_m,
                        source_surf_nodes=center_nodes_)
                qe = SurfaceVerticesQueryEngine(sel)
                sl = Searchlight(m, queryengine=qe)
                r = sl(ds).samples

            elif call_method_ == 'gam':
                # build everything from the ground up
                vg = volgeom.from_any(volume_,
                                      volume_mask_)
                vs = volsurf.VolSurfMaximalMapping(vg, s_i, s_o)
                sel = surf_voxel_selection.voxel_selection(
                        vs, radius, source_surf=s_m,
                        source_surf_nodes=center_nodes_)
                mp = sel

                ks = sel.keys()
                nk = len(ks)
                r = np.zeros((1, nk))
                for i, k in enumerate(ks):
                    r[0, i] = len(mp[k])

            # check if result is as expected
            assert_array_equal(r_expected, r)

        # clean up
        all_fns = [volfn, volfngz, outerfn, innerfn, intermediatefn]
        map(os.remove, all_fns)

        for k, vs in params.iteritems():
            if not k in tested_params:
                raise ValueError("Missing key: %r" % k)
            for v in vs:
                vstr = val2str(v)
                if not vstr in tested_params[k]:
                    raise ValueError("Missing value %r for %s" %
                                        (tested_params[k], k))


    def test_volsurf_projections(self):
        white = surf.generate_plane((0, 0, 0), (0, 1, 0), (0, 0, 1), 10, 10)
        pial = white + np.asarray([[1, 0, 0]])

        above = pial + np.asarray([[3, 0, 0]])
        vg = volgeom.VolGeom((10, 10, 10), np.eye(4))
        vs = volsurf.VolSurfMaximalMapping(vg, white, pial)

        dx = pial.vertices - white.vertices

        for s, w in ((white, 0), (pial, 1), (above, 4)):
            xyz = s.vertices
            ws = vs.surf_project_weights(True, xyz)
            delta = vs.surf_unproject_weights_nodewise(ws) - xyz
            assert_array_equal(delta, np.zeros((100, 3)))
            assert_true(np.all(w == ws))


        vs = volsurf.VolSurfMaximalMapping(vg, white, pial, nsteps=2)
        n2vs = vs.get_node2voxels_mapping()
        assert_equal(n2vs, dict((i, {i:0., i + 100:1.}) for i in xrange(100)))

        nd = 17
        ds_mm_expected = np.sum((above.vertices - pial.vertices[nd, :]) ** 2,
                                                                    1) ** .5
        ds_mm = vs.coordinates_to_grey_distance_mm(nd, above.vertices)
        assert_array_almost_equal(ds_mm_expected, ds_mm)

        ds_mm_nodewise = vs.coordinates_to_grey_distance_mm(True,
                                                            above.vertices)

        assert_array_equal(ds_mm_nodewise, np.ones((100,)) * 3)


    @with_tempfile('.h5py', 'voxsel')
    def test_surface_outside_volume_voxel_selection(self, fn):
        skip_if_no_external('h5py')
        from mvpa2.base.hdf5 import h5save, h5load
        vol_shape = (10, 10, 10, 1)
        vol_affine = np.identity(4)
        vg = volgeom.VolGeom(vol_shape, vol_affine)

        # make surfaces that are far away from all voxels
        # in the volume
        sphere_density = 4
        far = 10000.
        outer = surf.generate_sphere(sphere_density) * 10 + far
        inner = surf.generate_sphere(sphere_density) * 5 + far

        vs = volsurf.VolSurfMaximalMapping(vg, inner, outer)
        radii = [10., 10] # fixed and variable radii

        outside_node_margins = [0, far, True]
        for outside_node_margin in outside_node_margins:
            for radius in radii:
                selector = lambda:surf_voxel_selection.voxel_selection(vs,
                                            radius,
                                            outside_node_margin=outside_node_margin)

                if type(radius) is int and outside_node_margin is True:
                    assert_raises(ValueError, selector)
                else:
                    sel = selector()
                    if outside_node_margin is True:
                        # it should have all the keys, but they should
                        # all be empty
                        assert_array_equal(sel.keys(), range(inner.nvertices))
                        for k, v in sel.iteritems():
                            assert_equal(v, [])
                    else:
                        assert_array_equal(sel.keys(), [])

                    if outside_node_margin is True and \
                                 externals.versions['hdf5'] < '1.8.7':
                        raise SkipTest("Versions of hdf5 before 1.8.7 have "
                                                    "problems with empty arrays")

                    h5save(fn, sel)
                    sel_copy = h5load(fn)

                    assert_array_equal(sel.keys(), sel_copy.keys())
                    for k in sel.keys():
                        assert_equal(sel[k], sel_copy[k])

                    assert_equal(sel, sel_copy)



    def test_surface_voxel_query_engine(self):
        vol_shape = (10, 10, 10, 1)
        vol_affine = np.identity(4)
        vol_affine[0, 0] = vol_affine[1, 1] = vol_affine[2, 2] = 5
        vg = volgeom.VolGeom(vol_shape, vol_affine)

        # make the surfaces
        sphere_density = 10

        outer = surf.generate_sphere(sphere_density) * 25. + 15
        inner = surf.generate_sphere(sphere_density) * 20. + 15

        vs = volsurf.VolSurfMaximalMapping(vg, inner, outer)

        radius = 10

        for fallback, expected_nfeatures in ((True, 1000), (False, 183)):
            voxsel = surf_voxel_selection.voxel_selection(vs, radius)
            qe = SurfaceVoxelsQueryEngine(voxsel, fallback_euclidean_distance=fallback)

            # test i/o and ensure that the loaded instance is trained
            if externals.exists('h5py'):
                fd, qefn = tempfile.mkstemp('qe.hdf5', 'test'); os.close(fd)
                h5save(qefn, qe)
                qe = h5load(qefn)
                os.remove(qefn)


            m = _Voxel_Count_Measure()

            sl = Searchlight(m, queryengine=qe)

            data = np.random.normal(size=vol_shape)
            img = nb.Nifti1Image(data, vol_affine)
            ds = fmri_dataset(img)

            sl_map = sl(ds)

            counts = sl_map.samples

            assert_true(np.all(np.logical_and(5 <= counts, counts <= 18)))
            assert_equal(sl_map.nfeatures, expected_nfeatures)



    @reseed_rng()
    def test_surface_minimal_voxel_selection(self):
        # Tests 'minimal' voxel selection.
        # It assumes that 'maximal' voxel selection works (which is tested
        # in other unit tests)
        vol_shape = (10, 10, 10, 1)
        vol_affine = np.identity(4)
        vg = volgeom.VolGeom(vol_shape, vol_affine)

        # generate some surfaces,
        # and add some noise to them
        sphere_density = 10
        nvertices = sphere_density ** 2 + 2
        noise = np.random.uniform(size=(nvertices, 3))
        outer = surf.generate_sphere(sphere_density) * 5 + 8 + noise
        inner = surf.generate_sphere(sphere_density) * 3 + 8 + noise

        radii = [5., 20., 10] # note: no fixed radii at the moment

        # Note: a little outside margin is necessary
        # as otherwise there are nodes in the minimal case
        # that have no voxels associated with them

        for radius in radii:
            for output_modality in ('surface', 'volume'):
                for i, nvm in enumerate(('minimal', 'maximal')):
                    qe = disc_surface_queryengine(radius, vg, inner,
                                        outer, node_voxel_mapping=nvm,
                                        output_modality=output_modality)
                    voxsel = qe.voxsel

                    if i == 0:
                        keys_ = voxsel.keys()
                        voxsel_ = voxsel
                    else:
                        keys = voxsel.keys()
                        # minimal one has a subset
                        assert_equal(keys, keys_)

                        # and the subset is quite overlapping
                        assert_true(len(keys) * .90 < len(keys_))

                        for k in keys_:
                            x = set(voxsel_[k])
                            y = set(voxsel[k])

                            d = set.symmetric_difference(x, y)
                            r = float(len(d)) / 2 / len(x)
                            if type(radius) is float:
                                assert_equal(x - y, set())

                            # decent agreement in any case between the two sets
                            assert_true(r < .6)

    @reseed_rng()
    @with_tempfile('.h5py', 'voxsel')
    def test_queryengine_io(self, fn):
        skip_if_no_external('h5py')
        from mvpa2.base.hdf5 import h5save, h5load

        vol_shape = (10, 10, 10, 1)
        vol_affine = np.identity(4)
        vg = volgeom.VolGeom(vol_shape, vol_affine)

        # generate some surfaces,
        # and add some noise to them
        sphere_density = 10
        outer = surf.generate_sphere(sphere_density) * 5 + 8
        inner = surf.generate_sphere(sphere_density) * 3 + 8
        radius = 5.

        add_fa = ['center_distances', 'grey_matter_position']
        qe = disc_surface_queryengine(radius, vg, inner, outer,
                            add_fa=add_fa)
        ds = fmri_dataset(vg.get_masked_nifti_image())

        # the following is not really a strong requirement. XXX remove?
        assert_raises(ValueError, lambda: qe[qe.ids[0]])

        # check that after training it behaves well
        qe.train(ds)
        i = qe.ids[0]
        try:
            m = qe[i]
        except ValueError, e:
            raise AssertionError(
                'Failed to query %r from %r after training on %r. Exception was: %r'
                 % (i, qe, ds, e))

        assert_equal(qe[qe.ids[0]].samples[0, 0], 883)

        voxsel = qe.voxsel

        # store the original methods
        setstate_current = VolumeMaskDictionary.__dict__['__setstate__']
        reduce_current = VolumeMaskDictionary.__dict__['__reduce__']

        # try all combinations.
        # end with both set to False so that VolumeMaskDictionary is back
        # in its original state
        # XXX is manipulating class methods this way too dangerous?
        true_false_combis = [(i % 2 == 1, i // 2 == 0) for i in xrange(3, 7)]

        # try different ways to load volume mask dictionaries
        # first argument is filename, second argument is volume mask dictionary
        vmd_load_methods = [lambda f, vmd: h5load(f),
                            lambda f, vmd: volume_mask_dict.from_any(vmd),
                            lambda f, vmd: volume_mask_dict.from_any(f),
                            lambda f, vmd: vmd]
        for setstate_use_legacy, reduce_use_legacy in true_false_combis:
            reducer = VolumeMaskDictionary._reduce_legacy \
                            if reduce_use_legacy  \
                                else reduce_current
            VolumeMaskDictionary.__reduce__ = reducer

            setstater = VolumeMaskDictionary._setstate_legacy \
                            if setstate_use_legacy \
                            else setstate_current
            VolumeMaskDictionary.__setstate__ = setstater

            indices_stored = voxsel.__reduce__()[2][3]

            if reduce_use_legacy:
                assert_equal(type(indices_stored), dict)
                assert_equal(len(indices_stored), len(qe.ids))
            else:
                assert_equal(type(indices_stored), tuple)
                assert_equal(len(indices_stored), 3)
                for ix in indices_stored:
                    assert_equal(type(ix), np.ndarray)
            h5save(fn, qe)

            qe_copy = h5load(fn)

            if setstate_use_legacy and not reduce_use_legacy:
                assert_raises(AttributeError, lambda: qe_copy.ids)
                continue

            # ensure keys are the same
            assert_equal(qe.ids, qe_copy.ids)

            # ensure values are the same and that qe_copy is trained
            for id in qe.ids:
                assert_array_equal(qe[id].samples, qe_copy[id].samples)

            sel = qe.voxsel
            h5save(fn, sel)
            for vmd_load_method in vmd_load_methods:
                sel_copy = vmd_load_method(fn, sel)
                assert_equal(sel.aux_keys(), add_fa)
                expected_values = [1.13851869106, 1.03270423412] # smoke test
                for key, v in zip(add_fa, expected_values):
                    for id in qe.ids:
                        assert_array_equal(sel.get_aux(id, key), sel_copy.get_aux(id, key))

                    assert_array_almost_equal(sel.get_aux(qe.ids[0], key)[3], v)


    @with_tempfile('.h5py', 'voxsel')
    def test_surface_minimal_lowres_voxel_selection(self, fn):
        vol_shape = (4, 10, 10, 1)
        vol_affine = np.identity(4)
        vg = volgeom.VolGeom(vol_shape, vol_affine)


        # make surfaces that are far away from all voxels
        # in the volume
        sphere_density = 10
        radius = 10

        outer = surf.generate_plane((0, 0, 4), (0, .4, 0), (0, 0, .4), 14, 14)
        inner = outer + 2

        source = surf.generate_plane((0, 0, 4), (0, .8, 0), (0, 0, .8), 7, 7) + 1

        for i, nvm in enumerate(('minimal', 'minimal_lowres')):
            qe = disc_surface_queryengine(radius, vg, inner,
                                      outer, source,
                                      node_voxel_mapping=nvm)

            voxsel = qe.voxsel
            if i == 0:
                voxsel0 = voxsel
            else:
                assert_equal(voxsel.keys(), voxsel0.keys())
                for k in voxsel.keys():
                    p = voxsel[k]
                    q = voxsel0[k]

                    # require at least 60% agreement
                    delta = set.symmetric_difference(set(p), set(q))
                    assert_true(len(delta) < .8 * (len(p) + len(q)))

            if externals.exists('h5py'):
                from mvpa2.base.hdf5 import h5save, h5load

                h5save(fn, voxsel)
                voxsel_copy = h5load(fn)
                assert_equal(voxsel.keys(), voxsel_copy.keys())

                for id in qe.ids:
                    assert_array_equal(voxsel.get(id), voxsel_copy.get(id))




    @reseed_rng()
    def test_minimal_dataset(self):
        vol_shape = (10, 10, 10, 3)
        vol_affine = np.identity(4)
        vg = volgeom.VolGeom(vol_shape, vol_affine)

        data = np.random.normal(size=vol_shape)
        msk = np.ones(vol_shape[:3])
        msk[:, 1:-1:2, :] = 0

        ni_data = nb.Nifti1Image(data, vol_affine)
        ni_msk = nb.Nifti1Image(msk, vol_affine)

        ds = fmri_dataset(ni_data, mask=ni_msk)

        sphere_density = 20
        outer = surf.generate_sphere(sphere_density) * 10. + 5
        inner = surf.generate_sphere(sphere_density) * 7. + 5


        radius = 10
        sel = surf_voxel_selection.run_voxel_selection(radius, ds, inner, outer)


        sel_fids = set.union(*(set(sel[k]) for k in sel.keys()))

        ds_vox = map(tuple, ds.fa.voxel_indices)

        vg = sel.volgeom
        sel_vox = map(tuple, vg.lin2ijk(np.asarray(list(sel_fids))))


        fid_mask = np.asarray([v in sel_vox for v in ds_vox])
        assert_array_equal(fid_mask, sel.get_dataset_feature_mask(ds))

        # check if it raises errors
        ni_neg_msk = nb.Nifti1Image(1 - msk, vol_affine)
        neg_ds = fmri_dataset(ni_data, mask=ni_neg_msk) # inverted mask

        assert_raises(ValueError, sel.get_dataset_feature_mask, neg_ds)

        min_ds = sel.get_minimal_dataset(ds)
        assert_array_equal(min_ds.samples, ds[:, fid_mask].samples)


def _cartprod(d):
    '''makes a combinatorial explosion from a dictionary
    only combinations are made from values in tuples'''
    if not d:
        return [dict()]

    r = []
    k, vs = d.popitem()
    itervals = vs if type(vs) is tuple else [vs]
    for v in itervals:
        cps = _cartprod(d)
        for cp in cps:
            kv = {k:v}
            kv.update(cp)
            r.append(kv)
    d[k] = vs
    return r

class _Voxel_Count_Measure(Measure):
    # used to check voxel selection results
    is_trained = True
    def __init__(self, **kwargs):
        Measure.__init__(self, **kwargs)

    def _call(self, dset):
        return dset.nfeatures

def suite():  # pragma: no cover
    """Create the suite"""
    return unittest.makeSuite(SurfVoxelSelectionTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
