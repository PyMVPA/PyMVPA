# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
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
        intermediate: surf.Surface (default: None)
            Surface representing intermediate surface. If omitted
            it is the node-wise average of white and pial. 
            
        Notes
        -----
        'pial' and 'white' should have the same topology. 
        '''
        self._volgeom = volgeom.from_any(vg)
        self._pial = surf.from_any(pial)
        self._white = surf.from_any(white)

        if not self._pial.same_topology(self._white):
            raise Exception("Not same topology for white and pial")

        if intermediate is None:
            intermediate = (self.pial_surface * .5) + (self.white_surface * .5)

        self._intermediate = intermediate

    def __repr__(self, prefixes=[]):
        prefixes_ = ['vg=%r' % self._volgeom,
                     'white=%r' % self._white,
                     'pial=%r' % self._pial] + prefixes
        return "%s(%s)" % (self.__class__.__name__, ', '.join(prefixes_))


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

    def node2voxels(self, nsteps=10, start_fr=0.0,
                    stop_fr=1.0, start_mm=0, stop_mm=0):
        '''
        Generates a mapping from node indices to voxels that are at or near
        the nodes at the pial and white surface.
        
        Parameters
        ----------
        nsteps: int (default: 10)
            Number of nsteps from pial to white matter. For each node pair
            across the 'pial' and 'white' surface, a line is constructed 
            connecting the pairs. Subsequently 'nsteps' nsteps are taken from 
            'pial' to 'white' and the voxel at that position at the line is 
            associated with that node pair.
        start: float (default: 0)
            Relative start position of line in gray matter, 0.=white surface, 
            1.=pial surface
        stop: float (default: 1)
            Relative stop position of line, in gray matter, 0.=white surface, 
            1.=pial surface
            
              
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
        if start_fr > stop_fr or nsteps < 1:
            raise ValueError("Illegal start/stop combination, "
                             "or not enough steps")

        # make a list of the different relative gray matter positions
        if nsteps > 1:
            step = (stop_fr - start_fr) / float(nsteps - 1)
        else:
            step = 0.
            start_fr = stop_fr = .5

        # node to voxels mapping
        # if n2v[i]=vs, then node i is associated with the voxels vs
        #
        # vs is a mapping from indices to relative position in grey matter
        # where 0 means white surface and 1 means pial surface
        # vs[k]=pos means that voxel with linear index k is 
        # associated with relative positions pos0
        #
        # CHECKME that I did not confuse (the order of) pial and white surface

        center_ids = range(self._pial.nvertices)
        nv = len(center_ids) # number of nodes on the surface
        n2vs = dict() # node to voxel indices mapping
        for j in xrange(nv):
            n2vs[j] = None # by default, no voxels associated with each node

        vg = self._volgeom

        same_surfaces = self.white_surface == self.pial_surface

        surf_start = self.white_surface + start_mm
        surf_stop = self.pial_surface + stop_mm

        # different 'layers' (depths) in the grey matter
        for i in xrange(nsteps):
            whiteweight = start_fr + step * float(i) # ensure float
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

            for center_id in center_ids: # for each node on the surface
                # associate voxels with the present center node.
                # If a node is not in the volume, then no voxels are
                # associated with it. 

                if is_vox_in_vol[center_id]:
                    # no voxels (yet) associated with this node - make space
                    if n2vs[center_id] is None:
                        n2vs[center_id] = dict()

                    n2vs[center_id][lin_vox[center_id]] = \
                                                grey_matter_pos[center_id]

        return n2vs

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

        pxyz = self._pial.vertices
        weights = self.surf_project_weights_nodewise(xyz)
        return pxyz + np.reshape(weights, (-1, 1)) * pxyz

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
            P values of relative grey matter positions, where 0=white surface 
            and 1=pial surface.
        '''

        # compute relative to pial_xyz
        pxyz = self._pial.vertices
        qxyz = self._white.vertices

        dxyz = qxyz - pxyz # difference vector

        scale = np.sum(dxyz * dxyz, axis=1)


        weights = np.zeros((self._pial.nvertices,), dtype=pxyz.dtype)

        nan_mask = scale == 0
        weights[nan_mask] = np.nan

        non_nan_mask = np.logical_not(nan_mask)
        ps = xyz - pxyz
        proj = np.sum(ps * dxyz, axis=1)

        weights[non_nan_mask] = proj[non_nan_mask] / scale[non_nan_mask]

        return weights

    def voxel_count_nifti_image(self, n2v=None):
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

        if n2v is None:
            n2v = self.node2voxels()

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


    def __reduce__(self):
        return (self.__class__, (self._vg,))

    
    def __eq__(self, other):
        if not isinstance(other, VolumeBasedSurface):
            return False
        return self._vg==other._vg

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
            mn = src_ijk.ravel() - max_extent
            mx = src_ijk.ravel() + max_extent

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

    return VolSurf(vg, vs, vs, vs)



