# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##'''
"""
Associate volume geometry with two surface meshes (typically pial and white
matter boundaries of the grey matter).

@author: nick
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals
from mvpa2.misc.surfing import volgeom
from mvpa2.support.nibabel import surf

from mvpa2.base.progress import ProgressBar
from mvpa2.base import debug

if externals.exists('nibabel'):
        import nibabel as nb

class VolSurf(object):
    '''
    Associates a volume geometry with two surfaces (pial and white).
    '''
    def __init__(self, vg, white, pial, intermediate=None):
        '''
        Parameters
        ----------
        volgeom: volgeom.VolGeom
            Volume geometry
        white: surf.Surface
            Surface representing white-grey matter boundary
        pial: surf.Surface
            Surface representing pial-grey matter boundary
        intermediate: surf.Surface (default: None).
            Surface representing intermediate surface. If omitted
            it is the node-wise average of white and pial.
            This parameter is usually ignored, except when used
            in a VolSurfMinimalLowresMapping.

        Notes
        -----
        'pial' and 'white' should have the same topology.
        '''
        self._volgeom = volgeom.from_any(vg)
        self._pial = surf.from_any(pial)
        self._white = surf.from_any(white)

        if not self._pial.same_topology(self._white):
            raise Exception("Not same topology for white and pial")

        #if intermediate is None:
        #    intermediate = (self.pial_surface * .5) + (self.white_surface * .5)
        self._intermediate = surf.from_any(intermediate)


    def __repr__(self, prefixes=[]):
        prefixes_ = ['vg=%r' % self._volgeom,
                     'white=%r' % self._white,
                     'pial=%r' % self._pial] + prefixes
        return "%s(%s)" % (self.__class__.__name__, ', '.join(prefixes_))

    def __str__(self):
        return '%s(volgeom=%s, pial=%s, white=%s)' % (
                                            self.__class__.__name__,
                                            self._volgeom,
                                            self._white,
                                            self._pial)

    @property
    def pial_surface(self):
        '''
        Returns the pial surface

        Returns
        -------
        pial: surf.Surface
        '''
        return self._pial

    @property
    def white_surface(self):
        '''
        Returns the white surface

        Returns
        -------
        white: surf.Surface
        '''

        return self._white

    @property
    def intermediate_surface(self):
        '''
        Returns the node-wise average of the pial and white surface

        Returns
        -------
        intermediate: surf.Surface
        '''
        return self._intermediate

    @property
    def volgeom(self):
        '''
        Returns the volume geometry

        Returns
        -------
        vg: volgeom.VolGeom
        '''

        return self._volgeom

    def __reduce__(self):
        return (self.__class__, (self._volgeom, self._white,
                                 self._pial, self._intermediate))


    def surf_project_nodewise(self, xyz):
        '''
        Projects coordinates on lines connecting pial and white matter.

        Parameters
        ----------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial'
            surfaces have P nodes each

        Returns
        -------
        xyz_proj: numpy.ndarray (float)
            Px3 array with coordinates the constraints that xyz_proj[i,:]
            lies on the line connecting node 'i' on the white and pial
            surface, and that xyz_proj[i,:] is closest to xyz[i,:] of all
            points on this line.
        '''


        pxyz = self.white_surface.vertices
        weights = self.surf_project_weights_nodewise(xyz)
        return pxyz + np.reshape(weights, (-1, 1)) * pxyz

    def surf_unproject_weights_nodewise(self, weights):
        '''Maps relative positions in grey matter to coordinates

        Parameters
        ----------
        weights: numpy.ndarray (float)
            P values of relative grey matter positions, where 0=white surface
            and 1=pial surface.

        Returns
        -------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial' surfaces
            have P nodes each.
        '''
        # compute relative to pial_xyz
        pxyz = self._pial.vertices
        qxyz = self._white.vertices

        dxyz = (pxyz - qxyz)  # difference vector


        if len([s for s in weights.shape if s > 1]) != 1:
            raise ValueError("Weights should be a vector, but found "
                             "shape %s" % (weights.shape,))

        weights_lin = np.reshape(weights.ravel(), (-1, 1))
        return qxyz + weights_lin * dxyz


    def surf_project_weights_nodewise(self, xyz):
        '''
        Computes relative position of xyz on lines from pial to white matter.

        Parameters
        ----------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial' surfaces
            have P nodes each.

        Returns
        -------
        weights: numpy.ndarray (float)
            If nodes is True, P values of relative grey matter positions
            (0=white surface and 1=pial surface), where
            the i-th element are the projection weights for xyz[i] relative
            to the i-th node in the pial and white surface.
            Otherwise it returns an PxQ array with the projected weights
            for each node. If nodes is an int, then a P-vector is returned

        '''
        return self.surf_project_weights(True, xyz)


    def surf_project_weights(self, nodes, xyz):
        '''
        Computes relative position of xyz on lines from pial to white matter.

        Parameters
        ----------
        nodes: True or np.ndarray or int
            Q node indices for each the weights are computed. If True, then
            weights are computed node-wise, otherwise separately for each node.
        xyz: numpy.ndarray (float)
            Px3 array with coordinates. IF nodes is True then the 'white' and
            'pial' surfaces must have P nodes each.

        Returns
        -------
        weights: numpy.ndarray (float)
            If nodes is True, P values of relative grey matter positions
            (0=white surface and 1=pial surface), where
            the i-th element are the projection weights for xyz[i] relative
            to the i-th node in the pial and white surface.
            Otherwise it returns an PxQ array with the projected weights
            for each node. If nodes is an int, then a P-vector is returned

        '''
        node_wise = nodes is True

        if node_wise:
            one_node = True
            nodes = [False]  # placeholder
        else:
            one_node = type(nodes) is int
            nodes = np.asarray(nodes).ravel()
        nnodes = len(nodes)

        nxyz, three = xyz.shape
        if three != 3:
            raise ValueError('Coordinates should be Px3')


        pial = self._pial.vertices
        white = self._white.vertices

        weights = np.zeros((nxyz, nnodes))
        for i, node in enumerate(nodes):
            pxyz = pial if node_wise else pial[node, :]
            qxyz = white if node_wise else white[node, :]

            dxyz = pxyz - qxyz
            ndim = len(dxyz.shape)
            scale = np.sum(dxyz * dxyz, axis=ndim - 1)
            # weights = np.zeros((self._pial.nvertices,), dtype=pxyz.dtype)

            if node_wise:
                nan_mask = scale == 0
                weights[nan_mask, i] = np.nan

                non_nan_mask = np.logical_not(nan_mask)
                ps = (xyz - qxyz)
                proj = np.sum(ps * dxyz, axis=1)

                weights[non_nan_mask, i] = proj[non_nan_mask] / scale[non_nan_mask]
            else:
                if scale == 0:
                    weights[:, i] = np.NAN
                else:
                    ps = (xyz - qxyz)
                    proj = np.sum(ps * dxyz, axis=1)

                    weights[:, i] = proj / scale

        if one_node:
            weights = weights.ravel()

        return weights



    def coordinates_to_grey_distance_mm(self, nodes, xyz):
        '''Computes the grey position of coordinates in metric units

        Parameters
        ----------
        nodes: int or np.ndarray
            Single index, or Q indices of nodes relative to which the
            coordinates are computed. If True then grey distances
            are computed node-wise.
        xyz: Px3 array with coordinates, assuming 'white' and 'pial' surfaces
            have P nodes each.

        Returns
        -------
        grey_position_mm: np.ndarray
            Vector with P elements (if type(nodes) is int) or PxQ array
            (with type(nodes) is np.ndarray) containing the signed 'distance' to the
            grey matter. Values of zero indicate a node is within the grey
            matter. Negative values indicate that a node is 'below' the white
            matter (i.e. farther from the pial surface than the white surface),
            whereas Positive values indicate that a node is 'above' the pial
            matter.
        '''
        node_wise = nodes is True

        if node_wise:
            one_node = True
            all_nodes = [False]  # placeholder
        else:
            one_node = type(nodes) is int
            all_nodes = np.asarray(nodes).ravel()

        nnodes = len(all_nodes)

        nxyz, three = xyz.shape
        if three != 3:
            raise ValueError('Coordinates should be Px3')

        white = self.white_surface.vertices
        pial = self.pial_surface.vertices

        in_white = lambda x:x < 0
        in_pial = lambda x:x > 1

        # compute relative position
        pos = self.surf_project_weights(nodes, xyz)
        ds = np.zeros((nxyz, nnodes))  # space for output

        for i, node in enumerate(all_nodes):
            d = np.zeros(nxyz) + np.nan
            for sgn, s, f in ((-1, white, in_white), (1, pial, in_pial)):
                if node_wise:
                    msk = f(pos)
                    delta = s[msk] - xyz[msk] # difference in coordinates
                else:
                    # mask of voxels outside grey matter
                    msk = f(pos if one_node else pos[:, i])
                    delta = s[node, :] - xyz[msk, :]

                dst = np.sum(delta ** 2, 1) ** .5
                d[msk] = sgn * dst # compute signed distance
            d[np.isnan(d)] = 0
            ds[:, i] = d

        if one_node:
            ds = ds.ravel()

        return ds

class VolSurfMapping(VolSurf):
    '''General mapping between volume and surface.

    Subclasses have to implement node2voxels'''
    def __init__(self, vg, white, pial, intermediate=None,
                   nsteps=10, start_fr=0.0, stop_fr=1.0, start_mm=0, stop_mm=0):
        '''
        Parameters
        ----------
        volgeom: volgeom.VolGeom
            Volume geometry
        white: surf.Surface
            Surface representing white-grey matter boundary
        pial: surf.Surface
            Surface representing pial-grey matter boundary
        intermediate: surf.Surface (default: None).
            Surface representing intermediate surface. If omitted
            it is the node-wise average of white and pial.
        nsteps: int (default: 10)
            Number of steps from white to pial surface
        start_fr: float (default: 0)
            Relative start position of line in gray matter, 0.=white
            surface, 1.=pial surface.
        stop_fr: float (default: 1)
            Relative stop position of line (as in see start).
        start_mm: float (default: 0)
            Absolute start position offset (as in start_fr).
        stop_mm: float (default: 0)
            Absolute start position offset (as in start_fr).


        Notes
        -----
        'pial' and 'white' should have the same topology.
        '''
        super(VolSurfMapping, self).__init__(vg=vg, white=white, pial=pial,
                                             intermediate=intermediate)
        self.nsteps = nsteps
        self.start_fr = start_fr
        self.stop_fr = stop_fr
        self.start_mm = start_mm
        self.stop_mm = stop_mm


    def get_node2voxels_mapping(self):
        '''
        Returns
        -------
        n2v: dict
        A mapping from node indices to voxels. In this mapping, the
        'i'-th node is associated with 'n2v[i]=v2p' which contains the
        mapping from linear voxel indices to grey matter positions. In
        other words, 'n2v[i][idx]=v2p[idx]=pos' means that the voxel with
        linear index 'idx' is associated with node 'i' and has has
        relative position 'pos' in the gray matter.

        If node 'i' is outside the volume, then 'n2v[i]=None'.

        Notes
        -----
        The typical use case is selecting voxels in the grey matter. The
        rationale of this method is that (assuming a sufficient dense cortical
        surface mesh, combined with a sufficient number of nsteps, the grey
        matter is sampled dense enough so that 'no voxels are left out'.

        '''

        raise NotImplementedError

    def voxel_count_nifti_image(self):
        '''
        Returns a NIFTI image indicating how often each voxel is selected.

        Parameters
        ==========
        n2v: dict
            Node to voxel mapping, typically from node2voxels. If omitted
            then the output from node2voxels() is used.

        Returns
        =======
        img: nifti.Nifti1Image
            Image where the value in each voxel indicates how often
            each voxel was selected by n2v.
        '''

        n2v = self.get_node2voxels_mapping()

        v = self._volgeom
        nv = v.nvoxels

        voldata = np.zeros((nv,), dtype=float)

        for vx2d in n2v.itervalues():
            if vx2d:
                for vx in vx2d:
                    voldata[vx] += 1

        rs = np.reshape(voldata, v.shape)
        img = nb.Nifti1Image(rs, v.affine)
        return img

    def _get_node_voxels_maximal_mapping(self):
        '''Internal helper function to return all possible voxels associated
        with each node. Each voxel can be associated with multiple nodes

        It returns a node to voxel mapping and a voxel to node mapping'''

        nsteps = self.nsteps
        start_fr = self.start_fr
        stop_fr = self.stop_fr
        start_mm = self.start_mm
        stop_mm = self.stop_mm

        if start_fr > stop_fr or nsteps < 1:
            raise ValueError("Illegal start/stop combination, "
                             "or not enough steps")

        # make a list of the different relative gray matter positions
        if nsteps > 1:
            step = (stop_fr - start_fr) / float(nsteps - 1)
        else:
            step = 0.
            start_fr = stop_fr = .5

        center_ids = range(self._pial.nvertices)
        nv = len(center_ids)  # number of nodes on the surface

        vg = self._volgeom
        same_surfaces = self.white_surface == self.pial_surface

        surf_start = self.white_surface + start_mm
        surf_stop = self.pial_surface + stop_mm

        # allocate space for output
        # if n2v[i]=vs, then node i is associated with the voxels vs
        #
        # vs is a mapping from indices to relative position in grey matter
        # where 0 means white surface and 1 means pial surface
        # vs[k]=pos means that voxel with linear index k is
        # associated with relative positions pos0
        #
        # CHECKME that I did not confuse (the order of) pial and white surface
        #
        # v2ns is a mapping from voxel indices to sets of nodes.

        n2vs = dict()  # node to voxel indices mapping
        v2ns = dict()

        # by default, no voxels associated with each node
        for j in xrange(nv):
            n2vs[j] = None

        # different 'layers' (depths) in the grey matter
        for i in xrange(nsteps):
            whiteweight = start_fr + step * float(i)  # ensure float
            pialweight = 1 - whiteweight

            # compute weighted intermediate surface in between pial and white
            surf_weighted = surf_stop * pialweight + surf_start * whiteweight

            # coordinates
            surf_xyz = surf_weighted.vertices

            # linear indices of voxels containing nodes
            lin_vox = vg.xyz2lin(surf_xyz)

            # which of these voxels are actually in the volume
            is_vox_in_vol = vg.contains_lin(lin_vox)

            if same_surfaces:
                # prevent division by zero - simply assign it whatever weight is here
                grey_matter_pos = np.zeros(lin_vox.shape) + whiteweight
            else:
                # coordinates of voxels
                vol_xyz = vg.lin2xyz(lin_vox)

                # compute relative position of each voxel in grey matter
                grey_matter_pos = self.surf_project_weights_nodewise(vol_xyz)

            for center_id in center_ids:  # for each node on the surface
                # associate voxels with the present center node.
                # If a node is not in the volume, then no voxels are
                # associated with it.

                if is_vox_in_vol[center_id]:
                    # no voxels (yet) associated with this node - make space
                    if n2vs[center_id] is None:
                        n2vs[center_id] = dict()

                    vox_id = lin_vox[center_id]

                    # node to voxel mapping
                    n2vs[center_id][vox_id] = grey_matter_pos[center_id]

                    # voxel to node mapping
                    if not vox_id in v2ns:
                        v2ns[vox_id] = set()

                    v2ns[vox_id].add(center_id)

        return n2vs, v2ns

    def get_parameter_dict(self):
        '''
        Returns a dictionary with the most important parameters
        of this instance'''

        parameter_dict = dict(volgeom=self.volgeom,
                              volsurf_nvertices=self.white_surface.nvertices,
                              nsteps=self.nsteps,
                              start_fr=self.start_fr, stop_fr=self.stop_fr,
                              start_mm=self.start_mm, stop_mm=self.stop_mm)
        parameter_dict['class'] = self.__class__.__name__
        return parameter_dict



class VolSurfMaximalMapping(VolSurfMapping):
    def __init__(self, vg, white, pial, intermediate=None,
                   nsteps=10, start_fr=0.0,
                   stop_fr=1.0, start_mm=0, stop_mm=0):
        '''
        Represents the maximal mapping from nodes to voxels.
        'maximal', in this context, means that to each node all voxels
        are associated that are contained in lines connecting white
        and grey matter.

        Each voxel can be associated with multiple nodes.

        Parameters
        ----------
        volgeom: volgeom.VolGeom
            Volume geometry
        white: surf.Surface
            Surface representing white-grey matter boundary
        pial: surf.Surface
            Surface representing pial-grey matter boundary
        intermediate: surf.Surface (default: None).
            Surface representing intermediate surface. If omitted
            it is the node-wise average of white and pial.
        nsteps: int (default: 10)
            Number of steps from white to pial surface
        start_fr: float (default: 0)
            Relative start position of line in gray matter, 0.=white
            surface, 1.=pial surface.
        stop_fr: float (default: 1)
            Relative stop position of line (as in see start).
        start_mm: float (default: 0)
            Absolute start position offset (as in start_fr).
        stop_mm: float (default: 0)
            Absolute start position offset (as in start_fr).


        Notes
        -----
        'pial' and 'white' should have the same topology.
        '''

        super(VolSurfMaximalMapping, self).__init__(vg=vg, white=white, pial=pial,
                                intermediate=intermediate, nsteps=nsteps, start_fr=start_fr,
                                stop_fr=stop_fr, start_mm=start_mm, stop_mm=stop_mm)

    def get_node2voxels_mapping(self):
        '''Returns a mapping from nodes to voxels'''
        node2voxels_mapping, _ = self._get_node_voxels_maximal_mapping()
        return node2voxels_mapping


class VolSurfMinimalMapping(VolSurfMapping):
    def __init__(self, vg, white, pial, intermediate=None,
                   nsteps=10, start_fr=0.0,
                   stop_fr=1.0, start_mm=0, stop_mm=0):
        '''
        Represents the minimal mapping from nodes to voxels.
        'minimal', in this context, means that the mapping from
        voxels to nodes is many-to-one (i.e. each voxel is associated
        with at most one node)

        Each voxel can be associated with just a single node.

        Parameters
        ----------
        volgeom: volgeom.VolGeom
            Volume geometry
        white: surf.Surface
            Surface representing white-grey matter boundary
        pial: surf.Surface
            Surface representing pial-grey matter boundary
        intermediate: surf.Surface (default: None).
            Surface representing intermediate surface. If omitted
            it is the node-wise average of white and pial.
        nsteps: int (default: 10)
            Number of steps from white to pial surface
        start_fr: float (default: 0)
            Relative start position of line in gray matter, 0.=white
            surface, 1.=pial surface.
        stop_fr: float (default: 1)
            Relative stop position of line (as in see start).
        start_mm: float (default: 0)
            Absolute start position offset (as in start_fr).
        stop_mm: float (default: 0)
            Absolute start position offset (as in start_fr).

        Notes
        -----
        'pial' and 'white' should have the same topology.
        '''

        super(VolSurfMinimalMapping, self).__init__(vg=vg, white=white, pial=pial,
                                intermediate=intermediate, nsteps=nsteps, start_fr=start_fr,
                                stop_fr=stop_fr, start_mm=start_mm, stop_mm=stop_mm)


    def get_node2voxels_mapping(self):
        # start out with the maximum mapping, then prune it
        n2vs_max, v2ns_max = self._get_node_voxels_maximal_mapping()

        if __debug__ and 'SVS' in debug.active:
            nnodes = len(n2vs_max)
            nvoxels_max = sum(map(len, v2ns_max.itervalues()))
            nvoxels_max_per_node = float(nvoxels_max) / nnodes
            debug('SVS', 'Maximal node-to-voxel mapping: %d nodes, '
                            '%d voxels, %.2f voxels/node' %
                            (nnodes, nvoxels_max, nvoxels_max_per_node))
            debug('SVS', 'Starting injective pruning')


        # initialize mapping
        n2vs_min = dict((n, None if vs is None else dict())
                                    for n, vs in n2vs_max.iteritems())
        v2n_min = dict()

        # helper function to compute distance to intermediate surface
        dist_func = lambda (_, p): abs(p - .5)

        for v, ns in v2ns_max.iteritems():
            # get pairs os nodes and the voxel positions
            ns_pos = [(n, n2vs_max[n].get(v)) for n in ns]

            # get node nearest to intermediate surface
            min_node, min_pos = min(ns_pos, key=dist_func)


            if n2vs_min[min_node] is None:
                n2vs_min[min_node] = dict()
            assert(not v in n2vs_min[min_node]) # no duplicates
            n2vs_min[min_node][v] = min_pos

            assert(not v in v2n_min)
            v2n_min[v] = min_node

        if __debug__ and 'SVS' in debug.active:
            nvoxels_min = len(v2n_min)
            nvoxels_min_per_node = float(nvoxels_min) / nnodes
            nvoxels_delta = nvoxels_max - nvoxels_min
            nvoxels_pruned_ratio = float(nvoxels_delta) / nvoxels_max

            debug('SVS', 'Pruned %d/%d voxels (%.1f%%), %.2f voxels/node'
                             % (nvoxels_delta, nvoxels_max,
                                nvoxels_pruned_ratio * 100,
                                nvoxels_min_per_node))

        return n2vs_min

class VolSurfMinimalLowresMapping(VolSurfMinimalMapping):
    def __init__(self, vg, white, pial, intermediate=None,
                   nsteps=10, start_fr=0.0,
                   stop_fr=1.0, start_mm=0, stop_mm=0):
        '''
        Represents the minimal mapping from nodes to voxels,
        incorporating the intermediate surface that can
        be of lower-res.
        'minimal', in this context, means that the mapping from
        voxels to nodes is many-to-one (i.e. each voxel is associated
        with at most one node). Each node mapped must be
        present in the intermediate surface

        Each voxel can be associated with just a single node.

        Parameters
        ----------
        volgeom: volgeom.VolGeom
            Volume geometry
        white: surf.Surface
            Surface representing white-grey matter boundary
        pial: surf.Surface
            Surface representing pial-grey matter boundary
        intermediate: surf.Surface (default: None).
            Surface representing intermediate surface.
            Unlike in its superclass this argument cannot be ommited here.
        nsteps: int (default: 10)
            Number of steps from white to pial surface
        start_fr: float (default: 0)
            Relative start position of line in gray matter, 0.=white
            surface, 1.=pial surface.
        stop_fr: float (default: 1)
            Relative stop position of line (as in see start).
        start_mm: float (default: 0)
            Absolute start position offset (as in start_fr).
        stop_mm: float (default: 0)
            Absolute start position offset (as in start_fr).

        Notes
        -----
        'pial' and 'white' should have the same topology.
        '''

        if intermediate is None:
            raise RuntimeError("intermediate surface has to be specified")

        super(VolSurfMinimalLowresMapping, self).__init__(vg=vg, white=white,
                pial=pial, intermediate=intermediate, nsteps=nsteps,
                start_fr=start_fr, stop_fr=stop_fr, start_mm=start_mm,
                stop_mm=stop_mm)

    def get_node2voxels_mapping(self):
        n2v = super(VolSurfMinimalLowresMapping, self).\
                                get_node2voxels_mapping()

        # set low and high res intermediate surfaces
        lowres = surf.from_any(self._intermediate)
        highres = (self.pial_surface * .5) + \
                                (self.white_surface * .5)

        high2high_in_low = lowres.vonoroi_map_to_high_resolution_surf(highres)

        n_in_low2v = dict()
        ds = []

        for n, v2pos in n2v.iteritems():
            (n_in_low, d) = high2high_in_low[n]
            if v2pos is None:
                continue

            ds.append(d)


            if not n_in_low in n_in_low2v:
                # not there - just set the dictionary
                n_in_low2v[n_in_low] = v2pos
            else:
                # is there - see if it is none
                cur = n_in_low2v[n_in_low]
                if cur is None and not v2pos is None:
                    # also overwrite (v2pos can also be None, that's fine)
                    n_in_low2v[n_in_low] = v2pos
                elif v2pos is not None:
                    # update
                    for v, pos in v2pos.iteritems():
                        # minimal mapping, so voxel should not be there already
                        assert(not v in n_in_low2v[n_in_low])
                        cur[v] = pos

        if __debug__ and 'SVS' in debug.active:
            ds = np.asarray(ds)
            mu = np.mean(ds)
            n = len(ds)
            s = np.std(ds)

            debug('SVS', 'Reassigned %d nodes by moving %.2f +/- %.2f to low-res',
                        (n, mu, s))



        return n_in_low2v





class VolumeBasedSurface(surf.Surface):
    '''A surface based on a volume, where every voxel is a node.
    It has the empty topology, meaning there are no edges between
    nodes (voxels)

    Use case: provide volume-based searchlight behaviour. In that
    case finding neighbouring nodes is supposed to be faster
    using the circlearound_n2d method.

    XXX make a separate module?'''
    def __init__(self, vg):
        '''
        Parameters
        ----------
        vg: Volgeom.volgeom or str or NiftiImage
            volume to be used as a surface
        '''
        self._vg = volgeom.from_any(vg)

        n = self._vg.nvoxels
        vertices = self._vg.lin2xyz(np.arange(n))
        faces = np.zeros((0, 3), dtype=np.int)

        # call the parent's class constructor
        super(VolumeBasedSurface, self).__init__(vertices, faces, check=False)

    def __repr__(self, prefixes=[]):
        prefixes_ = ['vg=%r' % self._vg] + prefixes
        return "%s(%s)" % (self.__class__.__name__, ', '.join(prefixes_))

    def __str__(self):
        return '%s(volgeom=%s)' % (self.__class__.__name__,
                                     self._vg)

    def __reduce__(self):
        return (self.__class__, (self._vg,))


    def __eq__(self, other):
        if not isinstance(other, VolumeBasedSurface):
            return False
        return self._vg == other._vg

    def circlearound_n2d(self, src, radius, metric='euclidean'):
        shortmetric = metric[0].lower()

        if shortmetric == 'e':
            v = self.vertices

            # make sure src is a 1x3 array
            if type(src) is tuple and len(src) == 3:
                src = np.asarray(src)

            if isinstance(src, np.ndarray):
                if src.shape not in ((1, 3), (3,), (3, 1)):
                    raise ValueError("Illegal shape: should have 3 elements")

                src_coord = src if src.shape == (1, 3) else np.reshape(src, (1, 3))
            else:
                src_coord = np.reshape(v[src, :], (1, 3))

            # ensure it is a float
            src_coord = np.asanyarray(src_coord, dtype=np.float)

            # make a mask around center
            voxel2world = self._vg.affine
            world2voxel = np.linalg.inv(voxel2world)

            nrm = np.linalg.norm(voxel2world, 2)

            max_extent = np.ceil(radius / nrm + 1)

            src_ijk = self._vg.xyz2ijk(src_coord)

            # min and max ijk coordinates
            mn = (src_ijk.ravel() - max_extent).astype(np.int_)
            mx = (src_ijk.ravel() + max_extent).astype(np.int_)


            # set boundaries properly
            mn[mn < 0] = 0

            sh = np.asarray(self._vg.shape[:3])
            mx[mx > sh] = sh[mx > sh]

            msk_ijk = np.zeros(self._vg.shape[:3], np.int)
            msk_ijk[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]] = 1

            msk_lin = msk_ijk.ravel()

            # indices of voxels around the mask
            idxs = np.nonzero(msk_lin)[0]

            d = volgeom.distance(src_coord, v[idxs])[0, :]

            n = d.size
            node2dist = dict((idxs[i], d[i]) for i in np.arange(n)
                                    if d[i] <= radius)

            return node2dist

        elif shortmetric == 'd':
            return {src:0.}
        else:
            raise ValueError("Illegal metric: %s" % metric)


def from_volume(v):
    '''Makes a pseudo-surface from a volume.
    Each voxels corresponds to a node; there is no topology.
    A use case is mimicking traditional volume-based searchlights

    Parameters
    ----------
    v: str of NiftiImage
        input volume

    Returns
    -------
    s: surf.Surface
        Surface with an equal number as nodes as there are voxels
        in the input volume. The associated topology is empty.
    '''
    vg = volgeom.from_any(v)
    vs = VolumeBasedSurface(vg)

    return VolSurfMaximalMapping(vg, vs, vs, vs)



