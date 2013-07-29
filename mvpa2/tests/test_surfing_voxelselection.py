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
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

import nibabel as nb

import os
import tempfile

from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.datasets import Dataset
from mvpa2.measures.base import Measure
from mvpa2.datasets.mri import fmri_dataset

from mvpa2.support.nibabel import surf
from mvpa2.misc.surfing import surf_voxel_selection, queryengine, volgeom, \
                                volsurf

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


class SurfVoxelSelectionTests(unittest.TestCase):

    def test_voxel_selection(self):
        '''Compare surface and volume based searchlight'''

        '''
        Tests to see whether results are identical for surface-based
        searchlight (just one plane; Euclidian distnace) and volume-based
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
        vs = volsurf.VolSurf(vg, outer, inner)

        '''
        Run voxel selection with specified radius (in mm), using
        Euclidian distance measure
        '''
        surf_voxsel = surf_voxel_selection.voxel_selection(vs, radius,
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
                                            distance_metric='euclidian')
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
        fd, volfn = tempfile.mkstemp('vol.nii', 'test'); os.close(fd)
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
                vs = volsurf.VolSurf(vg, s_i, s_o)
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
                vs = volsurf.VolSurf(vg, s_i, s_o)
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
        vs = volsurf.VolSurf(vg, white, pial)

        dx = pial.vertices - white.vertices

        for s, w in ((white, 0), (pial, 1), (above, 4)):
            xyz = s.vertices
            ws = vs.surf_project_weights(True, xyz)
            delta = vs.surf_unproject_weights_nodewise(ws) - xyz
            assert_array_equal(delta, np.zeros((100, 3)))
            assert_true(np.all(w == ws))

        n2vs = vs.node2voxels()
        assert_equal(n2vs, dict((i, {i:0, i + 100:1}) for i in xrange(100)))

        nd = 17
        ds_mm_expected = np.sum((above.vertices - pial.vertices[nd, :]) ** 2,
                                                                    1) ** .5
        ds_mm = vs.coordinates_to_grey_distance_mm(nd, above.vertices)
        assert_array_almost_equal(ds_mm_expected, ds_mm)

        ds_mm_nodewise = vs.coordinates_to_grey_distance_mm(True,
                                                            above.vertices)
        assert_true(np.all(ds_mm_nodewise == 3))

    def test_surface_voxel_query_engine(self):
        vol_shape = (10, 10, 10, 1)
        vol_affine = np.identity(4)
        vol_affine[0, 0] = vol_affine[1, 1] = vol_affine[2, 2] = 5
        vg = volgeom.VolGeom(vol_shape, vol_affine)

        # make the surfaces
        sphere_density = 10

        outer = surf.generate_sphere(sphere_density) * 25. + 15
        inner = surf.generate_sphere(sphere_density) * 20. + 15

        vs = volsurf.VolSurf(vg, inner, outer)

        radius = 10

        for fallback, expected_nfeatures in ((True, 1000), (False, 183)):
            voxsel = surf_voxel_selection.voxel_selection(vs, radius)
            qe = SurfaceVoxelsQueryEngine(voxsel, fallback_euclidian_distance=fallback)

            m = _Voxel_Count_Measure()

            sl = Searchlight(m, queryengine=qe)

            data = np.random.normal(size=vol_shape)
            img = nb.Nifti1Image(data, vol_affine)
            ds = fmri_dataset(img)

            sl_map = sl(ds)

            counts = sl_map.samples

            assert_true(np.all(np.logical_and(5 <= counts, counts <= 18)))
            assert_equal(sl_map.nfeatures, expected_nfeatures)



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

def suite():
    """Create the suite"""
    return unittest.makeSuite(SurfVoxelSelectionTests)


if __name__ == '__main__':
    import runner
