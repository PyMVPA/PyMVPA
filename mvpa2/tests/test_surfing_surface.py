# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for cortical surface functionality"""

from mvpa2.testing.tools import skip_if_no_external

skip_if_no_external('matplotlib')
skip_if_no_external('griddata')

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import random

from mvpa2.testing.tools import assert_almost_equal, assert_array_equal, \
    assert_equal, assert_raises, assert_is_instance, unittest
from mvpa2.testing import sweepargs

from mvpa2.support.nibabel import surf
from mvpa2.support.nibabel.surf import vector_alignment_find_rotation, \
    generate_plane, Surface

from mvpa2.misc.plot.flat_surf import flat_surface2xy, FlatSurfacePlotter
from mvpa2.base.dataset import AttrDataset



def assert_vector_direction_almost_equal(x, y, *args, **kwargs):
    n_x = np.linalg.norm(x)
    n_y = np.linalg.norm(y)

    assert_almost_equal(np.dot(x, y), n_x * n_y, *args, **kwargs)



class SurfingSurfaceTests(unittest.TestCase):
    @staticmethod
    def _assert_rotation_maps_vector(r, x, y):
        # rotation must be 3x3 numpy array
        assert_equal(r.shape, (3, 3))
        assert_is_instance(r, np.ndarray)

        # rotation applied to x must yield direction of y
        # (modulo rounding errors)
        def normed(v):
            n_v = np.linalg.norm(v)

            return 0 if n_v == 0 else v / n_v

        rx = r.dot(x)

        rx_normed = normed(rx)
        y_normed = normed(y)
        assert_vector_direction_almost_equal(rx_normed, y_normed)

        # since it is a rotation, the result must have the same
        # L2 norm as the input
        assert_almost_equal(np.linalg.norm(x), np.linalg.norm(rx))

    def test_vector_alignment_find_rotation_random_vectors(self):
        x = np.random.normal(size=(3,))
        y = np.random.normal(size=(3,))

        r = vector_alignment_find_rotation(x, y)

        SurfingSurfaceTests._assert_rotation_maps_vector(r, x, y)

    def test_vector_alignment_find_rotation_canonical_vectors(self):
        for i in xrange(3):
            x = np.zeros((3,))
            x[i] = 1

            for j in xrange(3):
                y = np.zeros((3,))
                y[j] = 1

                r = vector_alignment_find_rotation(x, y)

                SurfingSurfaceTests._assert_rotation_maps_vector(r, x, y)

    def test_vector_alignment_find_rotation_illegal_inputs(self):
        arr = np.asarray
        illegal_args = [
            [arr([1, 2]), arr([1, 3])],
            [arr([1, 2, 3]), arr([1, 3])],
            [arr([1, 2, 3]), np.random.normal(size=(3, 3))]
        ]

        for illegal_arg in illegal_args:
            assert_raises((ValueError, IndexError),
                          vector_alignment_find_rotation, *illegal_arg)

    @sweepargs(do_deterioriate_surface=(False, True))
    def test_surface_face_normal(self, do_deterioriate_surface):
        vec1 = np.random.normal(size=(3,))
        vec2 = np.random.normal(size=(3,))

        vec_normal = -np.cross(vec1, vec2)

        plane = generate_plane((0, 0, 0), vec1, vec2, 10, 10)

        if do_deterioriate_surface:
            plane = SurfingSurfaceTests.deterioriate_surface(plane)

        plane_face_normals = plane.face_normals

        has_non_nan = False

        for f_n in plane_face_normals:
            if np.any(np.isnan(f_n)):
                continue

            assert_vector_direction_almost_equal(f_n, vec_normal, decimal=0)
            assert_almost_equal(f_n, surf.normalized(
                plane.nanmean_face_normal), decimal=0)

            has_non_nan = True

        if not has_non_nan:
            assert False, "Test should include faces with non-NaN normals"

    @staticmethod
    def add_noise_to_surface(s, noise_level=.05):
        vertices = s.vertices
        noise = np.random.uniform(size=vertices.shape, low=-.5 * noise_level,
                                  high=-.5 * noise_level, )
        vertices_noisy = vertices + np.random.uniform(size=vertices.shape) * \
                                    noise_level

        return Surface(vertices_noisy, s.faces, check=False)

    @staticmethod
    def set_nan_to_surface_vertices(s, nan_ratio=.05):
        # make some vertices NaN (as might be the case for flat surfaces)
        nan_count = int(np.ceil(s.nvertices * nan_ratio))
        nan_vertices_ids = np.random.random_integers(s.nvertices,
                                                     size=(nan_count,)) - 1
        vertices_noisy = s.vertices + 0.
        vertices_noisy[nan_vertices_ids, :] = np.nan

        return Surface(vertices_noisy, s.faces, check=False)

    @staticmethod
    def deterioriate_surface(s, noise_level=.05, nan_ratio=.05):
        s = SurfingSurfaceTests.add_noise_to_surface(s,
                                                     noise_level=noise_level)
        s = SurfingSurfaceTests.set_nan_to_surface_vertices(s,
                                                            nan_ratio=nan_ratio)
        return s

    @staticmethod
    def assert_coordinates_almost_equal_modulo_rotation(p_xyz, q_xyz,
                                                        max_difference):
        assert_equal(p_xyz.shape, q_xyz.shape)
        n, three = p_xyz.shape
        assert_equal(three, 3)

        n_pairs_to_test = 50

        get_random_int = lambda: int(random.uniform(0, n))
        get_distance = lambda x, y: np.linalg.norm(x - y)

        # ensure that we test for at least some distances, i.e.
        # that the presence of nans everywhere would not lead to a 'skipped'
        # test
        did_distance_test = False

        # compute some pairwise distances between nodes, and verity these
        # are more or lress the same in p_xyz and q_xyz
        for _ in xrange(n_pairs_to_test):
            a = get_random_int()
            b = get_random_int()

            d_p = get_distance(p_xyz[a], p_xyz[b])
            d_q = get_distance(q_xyz[a], q_xyz[b])

            if not any(np.isnan([d_p, d_q])):
                assert (abs(d_p - d_q) < max_difference)
                did_distance_test = True

        assert (did_distance_test)

    @sweepargs(dim=(0, 1, 2))
    def test_surface_flatten(self, dim):
        def unit_vec3(dim, scale):
            v = [0, 0, 0]
            v[dim] = float(scale)
            return tuple(v)

        origin = (0, 0, 0)
        plane_size = 10

        scale = 1.
        vec1 = unit_vec3(dim, scale=scale)
        vec2 = unit_vec3((dim + 1) % 3, scale=scale)

        plane = generate_plane(origin, vec1, vec2, plane_size, plane_size)

        noise_level = .05
        nan_vertices_ratio = .05

        # add some noise to spatial coordinates
        vertices = plane.vertices
        noise = np.random.uniform(size=vertices.shape,
                                  low=-.5,
                                  high=.5) * noise_level * scale
        vertices_noisy = vertices + noise

        # make some vertices NaN (as might be the case for flat surfaces)
        nan_count_float = plane.nvertices * nan_vertices_ratio
        nan_count = np.ceil(nan_count_float).astype(np.int)
        nan_vertices = np.random.random_integers(plane.nvertices,
                                                 size=(nan_count,)) - 1
        vertices_noisy[nan_vertices, dim] = np.nan
        plane_noisy = Surface(vertices_noisy, plane.faces)

        # compute normals
        f_normal = plane_noisy.face_normals

        # find average normal

        non_nan_f_normal = np.logical_not(np.any(np.isnan(f_normal), axis=1))
        f_normal_avg = np.mean(f_normal[non_nan_f_normal], axis=0)

        # test average normal
        assert_array_almost_equal(plane.nanmean_face_normal, f_normal_avg,
                                  decimal=2)

        # the output has only x and y coordinates; with z-coordinates set
        # to zero, the coordinates must be at similar pairwise distances
        max_deformation = .1
        x, y = flat_surface2xy(plane_noisy, max_deformation)
        n_vertices = plane.nvertices
        z = np.zeros((n_vertices,))
        flat_xyz = np.asarray((x, y, z))

        # nodes are rotated must have same pairwise distance as
        # the original surface
        max_difference = 3 * noise_level
        SurfingSurfaceTests.assert_coordinates_almost_equal_modulo_rotation(
            flat_xyz.T, plane.vertices, max_difference)

    def test_flat_surface_plotting(self):
        side = 10
        step = 1 / float(side)
        plane = surf.generate_plane((0, 0, 0), (step, 0, 0), (0, step, 0),
                                    side, side)

        # generate data with simple gradient
        data = plane.vertices[:, 0] - plane.vertices[:, 1]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        color_map = None

        img_side = 50
        fsp = FlatSurfacePlotter(plane, min_nsteps=img_side,
                                 color_map=color_map)
        img_arr = fsp(data)

        # verify shape
        ##assert_equal(img_arr.shape,(img_side,img_side,4))

        # get colors
        cmap = plt.get_cmap(color_map)
        expected_img_arr = cmap(data)

        # map vertex coordinates to indices in img_arr
        # (using nearest neighbor interpolation)
        xs = (plane.vertices[:, 0] * img_side).astype(np.int)
        ys = (plane.vertices[:, 1] * img_side).astype(np.int)

        # allocate space for rgb values
        img_rgb = np.zeros((plane.nvertices, 3))
        expected_img_rgb = np.zeros((plane.nvertices, 3))

        # store expected and found RGB values
        for i, (x, y) in enumerate(zip(*(xs, ys))):
            img_rgb[i] = img_arr[y, x, :3]
            expected_img_rgb[i] = expected_img_arr[i, :3]

        # RGB values should match
        c = np.corrcoef(img_rgb.T, expected_img_rgb.T)[:3, 3:6]
        assert (np.all(np.diag(c) > .9))

    def test_flat_surface_plotting_exception_wrong_size(self):
        s = surf.generate_plane((0, 0, 0), (0, 0, 1), (0, 1, 0), 6, 6)

        for offset in (-1, 0, 1):
            nfeatures = s.nvertices + offset
            ds = AttrDataset(samples=np.random.normal(size=(1, nfeatures)))

    def test_surfing_nodes_on_border_paths_surface_with_hole(self):
        s = surf.generate_plane((0, 0, 0), (0, 0, 1), (0, 1, 0), 6, 6)
        faces_to_remove = [1, 3, 7, 8, 3, 12, 13, 14, 22]
        faces_to_keep = np.setdiff1d(np.arange(s.nfaces), faces_to_remove)
        faces_to_add = [(0, 3, 10), (0, 4, 7), (0, 6, 4)]
        faces_hole = np.vstack((s.faces[faces_to_keep], faces_to_add))
        s_hole = surf.Surface(s.vertices, faces_hole)

        pths = s_hole.nodes_on_border_paths()

        expected_pths = [[1, 6, 4, 7, 0],
                         [3, 4, 9, 8, 2],
                         [11, 17, 23, 29, 35,
                          34, 33, 32, 31, 30,
                          24, 18, 12, 6, 7,
                          13, 19, 14, 9, 10,
                          5]]

        def as_sorted_sets(xs):
            return sorted(map(set, xs), key=min)

        assert_equal(as_sorted_sets(pths),
                     as_sorted_sets(expected_pths), )

    def test_average_node_edge_length(self):
        for side in xrange(1, 5):
            s_flat = surf.generate_plane((0, 0, 0), (0, 0, 1), (0, 1, 0), 6, 6)
            rnd_xyz = 0 * np.random.normal(size=s_flat.vertices.shape)
            s = surf.Surface(s_flat.vertices + rnd_xyz, s_flat.faces)

            nvertices = s.nvertices

            sd = np.zeros((nvertices,))
            c = np.zeros((nvertices,))

            def d(src, trg, vertices=s.vertices):
                s = vertices[src, :]
                t = vertices[trg, :]

                delta = s - t
                # print s, t, delta
                return np.sum(delta ** 2) ** .5

            for i_face in s.faces:
                for i in xrange(3):
                    src = i_face[i]
                    trg = i_face[(i + 1) % 3]

                    sd[src] += d(src, trg)
                    sd[trg] += d(src, trg)
                    c[src] += 1
                    c[trg] += 1

                    # print i, src, trg, d(src, trg)

            assert_array_almost_equal(sd / c, s.average_node_edge_length)

    def test_average_node_edge_length_tiny(self):
        a = np.random.uniform(low=2, high=5)
        b = np.random.uniform(low=2, high=5)
        c = (a ** 2 + b ** 2) ** .5

        vertices = [(0, 0, 0), (0, 0, a), (0, b, 0)]
        faces = [(0, 1, 2)]

        s = Surface(vertices, faces)
        expected_avg = [(a + b) / 2, (a + c) / 2, (b + c) / 2]
        assert_almost_equal(s.average_node_edge_length, expected_avg)
