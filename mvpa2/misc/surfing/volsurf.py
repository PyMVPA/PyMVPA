# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Associate volume geometry with surface meshes

@author: nick
"""

import surf, volgeom, sparse_attributes

import nibabel as nb
import numpy as np

class VolSurf():
    '''
    Associates a volume geometry with two surfaces (pial and white), so that 
    voxel selection in the volume can be performed using the surface information
    
    Parameters
    ----------
    volgeom: volgeom.VolGeom
        Volume geometry
    pial: surf.Surface
        Surface representing pial-grey matter boundary
    white: surf.Surface
        Surface representing white-grey matter boundary
    intermediate: surf.Surface
        Surface representing intermediate surface. This surface can have fewer 
        nodes than the pial and white surface. If None it uses the node-wise
         average of pial and white
    eps: float
        Maximum allowed distance between nodes on intermediate and the
        node-wise average of pial and white. This value is only used
        if intermediate is not None.
        
    Note
    ----
    'pial' and 'white' should have the same topology. 
    '''

    def __init__(self, vg, pial, white):
        self._volgeom = volgeom.from_any(vg)
        self._pial = surf.from_any(pial)
        self._white = surf.from_any(white)

        if not self._pial.same_topology(self._white):
            raise Exception("Not same topology for white and pial")

    def __repr__(self):
        r = ("volgeom: %r\npial: %rwhite:%r" %
                (self._volgeom, self._pial, self._white))
        return r

    def node2voxels(self, steps=10, start=0.0, stop=1.0):
        '''
        Generates a mapping from node indices to voxels that are at or near
        the nodes at the pial and white surface.
        
        Parameters
        ----------
        steps: int (default: 10)
            Number of steps from pial to white matter. For each node pair
            across the 'pial' and 'white' surface, a line is constructing connecting
            the pairs. Subsequently 'steps' steps are taken from 'pial' to 'white' 
            and the voxel at that position at the line is associated with that node pair.
        start: float (default: 0)
            Relative start position of line in gray matter, 0.=white surface, 1.=pial surface
            CheckMe: it might be the other way around 
        stop: float (default: 1)
            Relative stop position of line, in gray matter, 0.=white surface, 1.=pial surface
            
              
        Returns
        -------
        n2v: dict
            A mapping from node indices to voxels. In this mapping, the 'i'-th node 
            is associated with 'n2v[i]=v2p' which contains the mapping from 
            linear voxel indices to grey matter positions. In other words, 
            'n2v[i][idx]=v2p[idx]=pos' means that the voxel with linear index 'idx' is 
            associated with node 'i' and has has relative position 'pos' in the gray matter
            
            If node 'i' is outside the volume, then 'n2v[i]=None'
            
        Note
        ----
        The rationale is that a sufficient dense cortical surface mesh, combined
        with a sufficient number of steps, the grey matter is sampled dense enough
        so that  'no voxels are left out'. 
         
        '''
        if start > stop or steps < 1:
            raise ValueError("Illegal start/stop combination, or not enough steps")

        # make a list of the different relative gray matter positions
        if steps > 1:
            step = (stop - start) / float(steps - 1)
        else:
            step = 0
            start = stop = .5

        # node to voxels mapping
        # if n2v[i]=vs, then node i is associated with the voxels vs
        #
        # vs is a mapping from indices to relative position in grey matter
        # wheere 0 means white surface and 1 means pial surface
        # vs[k]=pos means that voxel with linear index k is 
        # associated with relative positions pos0
        #
        # CHECKME that I did not confuse (the order of) pial and white surface



        center_ids = range(self._pial.nvertices)
        nv = len(center_ids) # number of nodes on the surface
        n2vs = dict() # node to voxel indices mapping
        for j in xrange(nv):
            n2vs[j] = None # by default, no voxels associated with each node

        volgeom = self._volgeom

        # different 'layers' (depths) in the grey matter
        for i in xrange(steps):
            pialweight = start + step * i
            whiteweight = 1 - pialweight

            pialweight, whiteweight = whiteweight, pialweight
            # compute weighted intermediate surface in between pial and white
            surf = self._pial * pialweight + self._white * whiteweight

            surf_xyz = surf.vertices # coordinates

            lin_vox = volgeom.xyz2lin(surf_xyz) # linear indices of voxels containing nodes

            is_vox_in_vol = volgeom.contains_lin(lin_vox) # which of these voxels are actually in the volume

            vol_xyz = volgeom.lin2xyz(lin_vox) # coordinates of voxels

            # compute relative position of each voxel in grey matter
            grey_matter_pos = self.surf_project_weights_nodewise(vol_xyz)

            for center_id in center_ids: # for each node on the surface
                # associate voxels with the present center node
                # - center node must be in volume
                # - if onlnyinbetween is required, then also make sure that the 
                #   relative position is in between 0 and 1 (or whatever the values of
                #   start and stop are)

                if is_vox_in_vol[center_id]:
                    if n2vs[center_id] is None: # no voxels yet assocaited with this node
                        n2vs[center_id] = dict()

                    n2vs[center_id][lin_vox[center_id]] = grey_matter_pos[center_id]

        return n2vs

    def surf_project_nodewise(self, xyz):
        '''
        Project coordinates on lines connecting pial and white matter
        
        Parameters
        ----------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial' surfaces have P nodes each
        
        Returns
        -------
        xyz_proj: numpy.ndarray (float)
            Px3 array with coordinates the constraints that xyz_proj[i,:] lies on the 
            line connecting node 'i' on the white and pial surface, and that xyz_proj[i,:]
            is closest to xyz[i,:] of all points on this line.   
        '''

        pxyz = self._pial.vertices
        weights = self.surf_project_weights(xyz)
        return pxyz + np.reshape(weights, (-1, 1)) * pxyz

    def surf_project_weights_nodewise(self, xyz):
        '''
        Computes relative position of xyz on lines from pial to white matter
        
        Parameters
        ----------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial' surfaces have P nodes each
        
        Returns
        -------
        weights: numpy.ndarray (float)
            P values of relative grey matter positions, where 0=white surface and 1=pial surface
        '''

        # compute relative to pial_xyz
        pxyz = self._pial.vertices
        qxyz = self._white.vertices

        dxyz = qxyz - pxyz # difference vector

        scale = np.sum(dxyz * dxyz, axis=1)

        ps = xyz - pxyz
        proj = np.sum(ps * dxyz, axis=1)

        return proj / scale



    def voxel_count_image(self, n2v=None):
        '''
        Returns a NIFTI image indicating how often each voxel is selected
        
        Parameters
        ----------
        n2v: node to voxel mapping, typically from node2voxels. If omitted
            then the output from node2voxels() is used.
        
        Returns:
        img: nifti.Nifti1Image
            image where the value in each voxel indicates how often
            each voxel was selected
        '''

        v = self._volgeom
        nv = v.nvoxels

        voldata = np.zeros((nv,), dtype=float)

        for i, vx2d in n2v.iteritems():
            if vx2d:
                for vx in vx2d:
                    voldata[vx] += 1

        rs = np.reshape(voldata, v.shape)
        img = nb.Nifti1Image(rs, v.affine)
        return img
