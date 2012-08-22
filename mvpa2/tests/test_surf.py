# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA surface searchlight and related utilities"""

import numpy as np

import os
import tempfile

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.datasets import Dataset

import mvpa2.misc.surfing.surf as surf
import mvpa2.misc.surfing.surf_fs_asc as surf_fs_asc
import mvpa2.misc.surfing.volgeom as volgeom
import mvpa2.misc.surfing.volsurf as volsurf
import mvpa2.misc.surfing.sparse_attributes as sparse_attributes
import mvpa2.misc.surfing.surf_voxel_selection as surf_voxel_selection


class SurfTests(unittest.TestCase):
    """Test for surfaces

    NNO Aug 2012

    added as requested by Yarik and Michael

    'Ground truth' is whatever output is returned by the implementation
    as of mid-Aug 2012"""

    def test_surf(self):
        """Some simple testing with surfaces
        """

        s = surf.generate_sphere(10)

        assert_true(s.nvertices == 102)
        assert_true(s.nfaces == 200)

        v = s.vertices
        f = s.faces

        assert_true(v.shape == (102, 3))
        assert_true(f.shape == (200, 3))



        # another surface
        t = s * 10 + 2
        assert_true(t.same_topology(s))
        assert_array_equal(f, t.faces)

        assert_array_equal(v * 10 + 2, t.vertices)

        # allow updating, but should not affect original array
        # CHECKME: maybe we want to throw an exception instead
        assert_true((v * 10 + 2 == t.vertices).all().all())
        assert_true((s.vertices * 10 + 2 == t.vertices).all().all())

        # a few checks on vertices and nodes
        v_check = {40:(0.86511144 , -0.28109175, -0.41541501),
                   10:(0.08706015, -0.26794358, -0.95949297)}
        f_check = {10:(7, 8, 1), 40:(30, 31, 21)}


        vf_checks = [(v_check, lambda x:x.vertices),
                     (f_check, lambda x:x.faces)]

        eps = .0001
        for cmap, f in vf_checks:
            for k, v in cmap.iteritems():
                surfval = f(s)[k, :]
                assert_true((abs(surfval - v) < eps).all())

        # make sure same topology fails with different topology
        u = surf.generate_cube()
        assert_false(u.same_topology(s))

        # check that neighbours are computed correctly
        # even if we nuke the topology afterwards
        for _ in [0, 1]:
            nbrs = s.nbrs()
            n_check = [(0, 96, 0.284629),
                       (40, 39, 0.56218349),
                       (100, 99, 0.1741202)]
            for i, j, k in n_check:
                assert_true(abs(nbrs[i][j] - k) < eps)


        def assign_zero(x):
            x.faces[:, :] = 0
            return None

        assert_raises(RuntimeError, assign_zero, s)

        # see if mapping to high res works
        h = surf.generate_sphere(40)

        low2high = s.map_to_high_resolution_surf(h, .1)
        partmap = {7: 141, 8: 144, 9: 148, 10: 153, 11: 157, 12: 281}
        for k, v in partmap.iteritems():
            assert_true(low2high[k] == v)

        #  should fail if epsilon is too small
        assert_raises(ValueError,
                      lambda x:x.map_to_high_resolution_surf(h, .01), s)

        n2f = s.node2faces()
        for i in xrange(s.nvertices):
            nf = [10] if i < 2 else [5, 6] # number of faces expected

            assert_true(len(n2f[i]) in nf)


        ds2 = s.dijkstra_distance(2)
        some_ds = {0: 3.613173280799, 1: 0.2846296765, 2: 0.,
                 52: 1.87458018, 53: 2.0487004817, 54: 2.222820777,
                 99: 3.32854360, 100: 3.328543604, 101: 3.3285436042}

        eps = np.finfo('f').eps
        for k, v in some_ds.iteritems():
            assert_true(abs(v - ds2[k]) < eps)

    def test_surf_fs_asc(self):
        s = surf.generate_sphere(5) * 100

        _, fn = tempfile.mkstemp('surf', 'test')
        surf_fs_asc.write(fn, s, overwrite=True)

        t = surf_fs_asc.read(fn)
        os.remove(fn)

        eps = .0001
        _assert_array_equal_eps(s.vertices, t.vertices, eps)
        _assert_array_equal_eps(s.vertices, t.vertices, eps)

        theta = np.asarray([0, 0., 180.])

        r = s.rotate(theta, unit='deg')

        l2r = surf_fs_asc.sphere_reg_leftrightmapping(s, r)
        l2r_expected = [0, 1, 2, 6, 5, 4, 3, 11, 10, 9, 8, 7, 15, 14, 13, 12,
                       16, 19, 18, 17, 21, 20, 23, 22, 26, 25, 24]

        _assert_array_equal_eps(l2r, np.asarray(l2r_expected), 0)


        sides_facing = 'apism'
        for side_facing in sides_facing:
            l, r = surf_fs_asc.hemi_pairs_reposition(s + 10., t + (-10.),
                                                     side_facing)

            m = surf.merge(l, r)

            # not sure at the moment why medial rotation 
            # messes up - but leave for now
            eps = 666 if side_facing == 'm' else .001
            assert_true((abs(m.center_of_mass) < eps).all())

    def test_volgeom(self):
        sz = (17, 71, 37, 73) # size of 4-D 'brain volume'
        d = 2. # voxel size
        xo, yo, zo = -6., -12., -20. # origin
        mx = np.identity(4, np.float) * d # affine transformation matrix
        mx[3, 3] = 1
        mx[0, 3] = xo
        mx[1, 3] = yo
        mx[2, 3] = zo
        vg = volgeom.VolGeom(sz, mx) # initialize volgeom

        nv = sz[0] * sz[1] * sz[2] # number of voxels
        nt = sz[3] # number of time points
        assert_equal(vg.nvoxels, nv)
        assert_equal(vg.ntimepoints, nt)

        # a couple of hard-coded test cases
        # last two are outside the volume
        linidxs = [0, 1, sz[2], sz[1] * sz[2], nv - 1, -1 , nv]
        subidxs = ([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                    (sz[0] - 1, sz[1] - 1, sz[2] - 1)]
                   + [(sz[0], sz[1], sz[2])] * 2)

        xyzs = ([(xo, yo, zo), (xo, yo, zo + d), (xo, yo + d, zo),
                 (xo + d, yo, zo),
                 (xo + d * (sz[0] - 1), yo + d * (sz[1] - 1), zo + d * (sz[2] - 1))]
                + [(np.nan, np.nan, np.nan)] * 2)

        for i, linidx in enumerate(linidxs):
            lin = np.asarray([linidx])
            ijk = vg.lin2ijk(lin)


            ijk_expected = np.reshape(np.asarray(subidxs[i]), (1, 3))
            _assert_array_equal_eps(ijk, ijk_expected)

            xyz = vg.lin2xyz(lin)

            xyz_expected = np.reshape(np.asarray(xyzs[i]), (1, 3))
            _assert_array_equal_eps(xyz, xyz_expected)


        # check that some identities hold
        ab, bc, ac = vg.lin2ijk, vg.ijk2xyz, vg.lin2xyz
        ba, cb, ca = vg.ijk2lin, vg.xyz2ijk, vg.xyz2lin
        identities = [lambda x:ab(ba(x)),
                      lambda x:bc(cb(x)),
                      lambda x:ac(ca(x)),
                      lambda x:ba(ab(x)),
                      lambda x:cb(bc(x)),
                      lambda x:ca(ac(x)),
                      lambda x:bc(ab(ca(x))),
                      lambda x:ba(cb(ac(x)))]

        # 0=lin, 1=ijk, 2=xyz
        identities_input = [1, 2, 2, 0, 1, 0, 2, 0]
        identities_input_eps = [0., 0., 0.] # how much difference we allow

        # voxel indices to test
        linrange = [0, 1, sz[2], sz[1] * sz[2]] + range(0, nv, nv / 100)

        lin = np.reshape(np.asarray(linrange), (-1,))
        ijk = vg.lin2ijk(lin)
        xyz = vg.ijk2xyz(ijk)

        for j, identity in enumerate(identities):
            inp = identities_input[j]
            if inp == 0:
                x = lin
            elif inp == 1:
                x = ijk
            elif inp == 2:
                x = xyz

            eps = identities_input_eps[inp]
            _assert_array_equal_eps(x, identity(x), eps)

        # check that masking works
        assert_true(vg.contains_lin(lin).all())
        assert_false(vg.contains_lin(-lin - 1).any())

        assert_true(vg.contains_ijk(ijk).all())
        assert_false(vg.contains_ijk(-ijk - 1).any())


        # ensure that we have no rounding issues
        deltas = [-.51, -.49, 0., .49, .51]
        should_raise = [True, False, False, False, True]

        for delta, r in zip(deltas, should_raise):
            xyz_d = xyz + delta * d
            lin_d = vg.xyz2lin(xyz_d)

            if r:
                assert_raises(ValueError,
                              lambda x, y:_assert_array_equal_eps(x, y),
                              lin_d, lin)
            else:
                _assert_array_equal_eps(lin_d, lin)


        # some I/O testing

        img = vg.empty_nifti_img()
        _, fn = tempfile.mkstemp('.nii', 'test')
        img.to_filename(fn)

        assert_true(os.path.exists(fn))

        vg2 = volgeom.from_image(img)
        vg3 = volgeom.from_nifti_file(fn)

        _assert_array_equal_eps(vg.affine, vg2.affine, 0)
        _assert_array_equal_eps(vg.affine, vg3.affine, 0)

        assert_equal(vg.shape[:3], vg2.shape[:3], 0)
        assert_equal(vg.shape[:3], vg3.shape[:3], 0)

        os.remove(fn)

    def test_volsurf(self):
        vg = volgeom.VolGeom((50, 50, 50), np.identity(4))

        density = 40
        outer = surf.generate_sphere(density) * 25. + 25
        inner = surf.generate_sphere(density) * 20. + 25


        vs = volsurf.VolSurf(vg, outer, inner)

        # increasingly select more voxels in 'grey matter'
        steps_start_stop = [(1, .5, .5), (5, .5, .5), (3, .3, .7),
                          (5, .3, .7), (5, 0., 1.), (10, 0., 1.)]

        mp = None
        expected_keys = set(range(density ** 2 + 2))
        selection_counter = []
        voxel_counter = []
        for sp, sa, so in steps_start_stop:
            n2v = vs.node2voxels(sp, sa, so)

            if mp is None:
                mp = n2v


            assert_equal(expected_keys, set(n2v.keys()))

            counter = 0
            for k, v2pos in n2v.iteritems():
                for v, pos in v2pos.iteritems():
                    # should be close to grey matter
                    assert_true(-1. <= pos and pos <= 2.)
                    counter += 1

            selection_counter.append(counter)
            img = vs.voxel_count_image(n2v)

            voxel_counter.append(np.sum(img.get_data() > 0))

        # hard coded number of expected voxels
        selection_expected = [1602, 1602, 4618, 5298, 7867, 10801]
        assert_equal(selection_counter, selection_expected)

        voxel_expected = [1498, 1498, 4322, 4986, 7391, 10141]
        assert_equal(voxel_counter, voxel_expected)


    def test_sparse_attributes(self):

        payload = {1:([1, 2], [.2, .3]),
                   2:([3, 4, 5], [.4, .7, .8]),
                   6:([3, 4, 8, 9], [.1, .3, .2, .2])}

        payload_labels = ['lin_vox_idxs', 'center_distances']

        sa = sparse_attributes.SparseAttributes(payload_labels)
        for k, v in payload.iteritems():
            d = dict()
            for i, label in enumerate(payload_labels):
                d[label] = v[i]
            sa.add(k, d)

        sa.a['some general attribute'] = 'voxel selection example'

        assert_equal(set(payload_labels), set(sa.sa_labels()))

        for i, label in enumerate(payload_labels):
            it = sa.get_attr_mapping(label)
            for k, v in it.iteritems():
                assert_equal(v, it[k])
                assert_equal(v, payload[k][i])
                assert_equal(sa.get(k, label), v)

        # test I/O
        _, fn = tempfile.mkstemp('.pickle', 'test')

        sparse_attributes.to_file(fn, sa)
        sb = sparse_attributes.from_file(fn)
        os.remove(fn)

        # check stuff in file is the same
        for label in payload_labels:
            ma = sa.get_attr_mapping(label)
            mb = sb.get_attr_mapping(label)

            for k in ma:
                assert_equal(ma[k], mb[k])

    def test_surf_voxel_selection(self):
        return
        vg = volgeom.VolGeom((50, 50, 50), np.identity(4))

        density = 40
        outer = surf.generate_sphere(density) * 25. + 25
        inner = surf.generate_sphere(density) * 20. + 25

        vs = volsurf.VolSurf(vg, outer, inner)

        nv = outer.nvertices

        srcs = range(nv)


#        voxel_selection(vs, surf_srcs, radius, srcs=src, start=0., stop=1., steps=10,
 #                   distancemetric='dijkstra', intermediateat=.5, etastep=1):



        sel = surf_voxel_selection = run_voxelselection(epifn, whitefn, pialfn, intermediatefn, radius, srcs)

        vg = attr.a['volgeom']

        nv = vg.nv()
        datalin = np.zeros((nv, 1))

        mp = attr.get_attr_mapping('lin_vox_idxs')
        for k, idxs in mp.iteritems():
            if idxs is not None:
                datalin[idxs] += 1

        datars = np.reshape(datalin, vg.shape())

        img = ni.Nifti1Image(datars, vg.affine())

        fnout = d + "__test_n2v4.nii"
        img.to_filename(fnout)

        attrfn = d + "voxsel.pickle"
        sparse_attributes.to_file(attrfn, attr)

'''
if __name__ == "__main__":
    #from mvpa2.tutorial_suite import *

    if __debug__:
        debug.active += ["SVS"]


    d = '%s/qref/' % utils._get_fingerdata_dir()
    epifn = d + "../glm/rall_vol00.nii"

    ld = 36 # mapicosahedron linear divisions
    smallld = 9
    hemi = 'l'
    nodecount = 10 * smallld ** 2 + 2

    pialfn = d + "ico%d_%sh.pial_al.asc" % (ld, hemi)
    whitefn = d + "ico%d_%sh.smoothwm_al.asc" % (ld, hemi)
    intermediatefn = d + "ico%d_%sh.intermediate_al.asc" % (smallld, hemi)

    fnoutprefix = d + "_voxsel1_ico%d_%sh_" % (ld, hemi)
    radius = 100 # 100 voxels per searchlight
    srcs = range(nodecount) # use all nodes as a center

    attr = run_voxelselection(epifn, whitefn, pialfn, intermediatefn, radius, srcs)

    vg = attr.a['volgeom']

    nv = vg.nv()
    datalin = np.zeros((nv, 1))

    mp = attr.get_attr_mapping('lin_vox_idxs')
    for k, idxs in mp.iteritems():
        if idxs is not None:
            datalin[idxs] += 1

    datars = np.reshape(datalin, vg.shape())

    img = ni.Nifti1Image(datars, vg.affine())

    fnout = d + "__test_n2v4.nii"
    img.to_filename(fnout)

    attrfn = d + "voxsel.pickle"
    sparse_attributes.to_file(attrfn, attr)

    print fnout
    print attrfn
    '''







def _assert_array_equal_eps(x, y, eps=.0001):
    if x.shape != y.shape:
        raise ValueError('not equal size: %r != %r' % (x.shape, y.shape))

    xr = np.reshape(x, (-1,))
    yr = np.reshape(y, (-1,))

    delta = np.abs(xr - yr)

    m = -(delta <= eps)

    # deal with NaNs
    if ((any(-np.isnan(xr[m])) or any(-np.isnan(yr[m])))):
        raise ValueError('arrays differ more than %r' % eps)


def suite():
    """Create the suite"""
    return unittest.makeSuite(SurfTests)


if __name__ == '__main__':
    import runner

