# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Volume geometry to map between world and voxel coordinates.

Supports conversion between linear and sub indexing of voxels. The rationale is that
volumes use sub indexing that incorporate the spatial locations of voxels, but for 
voxel selection (and subsequent MVPA) we want to abstract from this spatial information 

Created on Feb 12, 2012

@author: nick
'''

import nibabel as nb, numpy as np, surf_fs_asc, utils

from mvpa2.datasets.mri import fmri_dataset

class VolGeom():
    '''
    Parameters
    ----------
    shape: tuple
        Number of items in each dimension.
        Typically the first three dimensions are spatial and the remaining ones 
        temporal.
    affine: numpy.ndarray
        4x4 affine transformation array that maps voxel to world coordinates.
    mask: numpy.ndarray (default: None)
        voxel mask that indicates which voxels are included. By default all 
        voxels are included.
    
    '''
    def __init__(self, shape, affine, mask=None):
        self._shape = shape
        self._affine = affine
        if not mask is None:
            # convert into linear numpy array of type boolean
            if not type(mask) is np.ndarray:
                raise ValueError("Mask not understood: not an numpy.ndarray")
            if mask.size != self.nvoxels:
                raise ValueError("%d voxels, but mask has %d" %
                                 (self.nvoxels, mask.size))
            mask = np.reshape(mask != 0, (-1,))
        self._mask = mask

    def as_pickable(self):
        '''
        Returns
        -------
        dict
            A dictionary that contains all information from this instance (and can be 
            saved using pickle)
        '''
        d = dict(shape=self.shape, affine=self.affine, mask=self.mask)
        return d

    @property
    def mask(self):
        '''
        Returns
        -------
        mask: np.ndarray
            boolean vector indicating which voxels are included
        '''
        if self._mask is None:
            return None

        m = self._mask.view()
        m.flags.writeable = False
        return m

    def _ijkmultfac(self):
        '''multiplication factors for ijk <--> linear indices conversion'''
        sh = self.shape
        return [sh[1] * sh[2], sh[2], 1]

    def _contains_ijk_unmasked(self, ijk):
        shape = self.shape

        m = reduce(np.logical_and, [0 <= ijk[:, 0], ijk[:, 0] < shape[0],
                                   0 <= ijk[:, 1], ijk[:, 1] < shape[1],
                                   0 <= ijk[:, 2], ijk[:, 2] < shape[2]])
        return m

    def _outside_vol(self, ijk, lin):
        invol = self._contains_ijk_unmasked(ijk)
        invol[np.logical_or(lin < 0, lin >= self.nvoxels)] = np.False_

        if not self.mask is None:
            #invol = np.logical_and(invol, self.mask[lin])
            invol[invol] = np.logical_and(invol[invol], self.mask[lin[invol]])

        return np.logical_not(invol)

    def _ijk2lin_unmasked(self, ijk):
        m = np.zeros((3,), dtype=int)
        fs = self._ijkmultfac()

        # make a 3x1 vector with multiplication factors
        for i, f in enumerate(fs):
            m[i] = f

        lin = np.dot(ijk, m)
        return lin

    def _lin2ijk(self, lin):
        '''Converts sub to linear voxel indices
        
        Parameters
        ----------
            Px1 array with linear voxel indices
            
        Returns
        -------
        ijk: numpy.ndarray
            Px3 array with sub voxel indices
        '''
        #outsidemsk = np.invert(self.contains_lin(lin))
        lin = lin.copy() # we'll change lin, don't want to mess with input
        #outsidemsk = np.logical_or(lin < 0, lin >= self.nvoxels)

        n = np.shape(lin)[0]
        fs = self._ijkmultfac()

        ijk = np.zeros((n, 3), dtype=int)
        for i, f in enumerate(fs):
            v = lin / f
            ijk[:, i] = v[:]
            lin -= v * f

        return ijk

    def ijk2lin(self, ijk):
        '''Converts sub to linear voxel indices
        
        Parameters
        ----------
        ijk: numpy.ndarray
            Px3 array with sub voxel indices
        
        Returns
        -------
            Px1 array with linear voxel indices
        '''

        lin = self._ijk2lin_unmasked(ijk)
        lin[self._outside_vol(ijk, lin)] = self.nvoxels

        return lin

    def lin2ijk(self, lin):
        '''Converts sub to linear voxel indices
        
        Parameters
        ----------
            Px1 array with linear voxel indices
            
        Returns
        -------
        ijk: numpy.ndarray
            Px3 array with sub voxel indices
        '''

        ijk = self._lin2ijk(lin)
        ijk[self._outside_vol(ijk, lin), :] = self.shape[:3]

        return ijk

    @property
    def affine(self):
        '''Returns the affine transformation matrix
        
        Returns
        -------
        affine : numpy.ndarray
            4x4 array that maps voxel to world coorinates
        '''

        a = self._affine.view()
        a.flags.writeable = False
        return a


    def xyz2ijk(self, xyz):
        '''Maps world to sub voxel coordinates
        
        Parameters
        ----------
        xyz : numpy.ndarray (float)
            Px3 array with world coordinates
        
        Returns
        -------
        ijk: numpy.ndarray (int)
            Px3 array with sub voxel indices.
            World coordinates outside the volume are set to 
            sub voxel indices outside the volume too
        '''
        m = self.affine
        minv = np.linalg.inv(m)

        ijkfloat = self.apply_affine3(minv, xyz)

        # add .5 so that positions are rounded instead of floored CHECKME
        ijk = np.array(ijkfloat + .5, dtype=int)

        lin = self._ijk2lin_unmasked(ijk)

        ijk[self._outside_vol(ijk, lin), :] = self.shape[:3]
        return ijk

    def ijk2xyz(self, ijk):
        '''Maps sub voxel indices to world coordinates
        
        Parameters
        ----------
        ijk: numpy.ndarray (int)
            Px3 array with sub voxel indices.
        
        Returns
        -------
        xyz : numpy.ndarray (float)
            Px3 array with world coordinates
            Voxel indices outside the volume are set to NaN
        '''

        m = self.affine
        ijkfloat = np.array(ijk, dtype=float)
        xyz = self.apply_affine3(m, ijkfloat)

        lin = self._ijk2lin_unmasked(ijk)
        self._outside_vol(ijk, lin)

        xyz[self._outside_vol(ijk, lin), :] = np.NaN
        return xyz


    def xyz2lin(self, xyz):
        '''Maps world to linear coordinates
        
        Parameters
        ----------
        xyz : numpy.ndarray (float)
            Px3 array with world coordinates
        
        Returns
        -------
        ijk: numpy.ndarray (int)
            Px1 array with linear indices.
            World coordinates outside the volume are set to 
            linear indices outside the volume too
        '''
        return self.ijk2lin(self.xyz2ijk(xyz))

    def lin2xyz(self, lin):
        '''Linear voxel indices to world coordinates
        
        Parameters
        ----------
        ijk: numpy.ndarray (int)
            Px3 array with linear voxel indices.
        
        Returns
        -------
        xyz : np.ndarray (float)
            Px1 array with world coordinates
            Voxel indices outside the volume are set to NaN
        '''

        return self.ijk2xyz(self.lin2ijk(lin))

    def apply_affine3(self, mat, v):
        '''Applies an affine transformation matrix
        
        Parameters
        ----------
        mat : numpy.ndarray (float)
            Matrix with size at least 3x4
        v : numpy.ndarray (float)
            Px3 values to which transformation is applied
        
        Returns
        -------
        w : numpy.ndarray(float)
            Px3 transformed values
        '''


        r = mat[:3, :3]
        t = mat[:3, 3].transpose()

        return np.dot(v, r) + t

    @property
    def ntimepoints(self):
        '''
        Returns
        -------
        int
            Number of time points
        '''
        return np.prod(self.shape[3:])
    @property
    def nvoxels(self):
        '''
        Returns
        -------
        int
            Number of spatial points (i.e. number of voxels)
        '''
        return np.prod(self.shape[:3])

    @property
    def shape(self):
        '''
        Returns
        -------
        tuple: int
            Number of values in each dimension'''


        return self._shape

    @property
    def nvoxels_mask(self):
        '''
        Returns
        -------
        int
            Number of voxels that survive the mask'''
        return self.nvoxels if self.mask is None else np.sum(self.mask)

    def contains_ijk(self, ijk):
        '''
        Parameters
        ----------
        ijk : numpy.ndarray
            Px3 array with sub voxel indices
        
        Returns
        -------
        numpy.ndarray (boolean)
            P boolean values indicating which voxels are within the volume
        '''

        #lin = self.ijk2lin(ijk)

        #return self.contains_lin(lin)

        shape = self.shape

        m = reduce(np.logical_and, [0 <= ijk[:, 0], ijk[:, 0] < shape[0],
                                   0 <= ijk[:, 1], ijk[:, 1] < shape[1],
                                   0 <= ijk[:, 2], ijk[:, 2] < shape[2]])

        if not self.mask is None:
            m = np.logical_and(m, self.mask)

        return m


    def contains_lin(self, lin):
        '''
        Parameters
        ----------
        lin : numpy.ndarray
            Px1 array with sub voxel indices
        
        Returns
        -------
        numpy.ndarray (boolean)
            P boolean values indicating which voxels are within the volume
        '''

        ijk = self.lin2ijk(lin)
        return self.contains_ijk(ijk)
        #nv = self.nvoxels
        #c = np.logical_and(0 <= lin, lin < nv)
        #if not self.mask is None:
        #    c[c] = self.mask[lin[c]]
        #return c

    def empty_nifti_img(self, nt=1):
        sh = self.shape
        sh4d = (sh[0], sh[1], sh[2], nt)

        data = np.zeros(sh4d)
        img = nb.Nifti1Image(data, self.affine)
        return img

    def masked_nifti_img(self, nt=1):
        data_lin = np.zeros(self.nvoxels, self.ntimepoints)
        if self.mask is None:
            data_lin[:] = 1
        else:
            data_lin[self.mask] = 1

        sh = self.shape
        sh4d = (sh[0], sh[1], sh[2], nt)

        data = np.reshape(data_lin, sh4d)
        img = nb.Nifti1Image(data, self.affine)
        return img


def from_nifti_file(fn, mask_volume_index=None):
    '''
    Parameters
    ----------
    fn : str
        Nifti filename
    mask_volume_index: int or None (default)
        which volume in fn to use as a voxel mask. None means no mask. 
    
    
    Returns
    -------
    vg: VolGeom
        Volume geometry associated with 'fn'
    '''

    img = nb.load(fn)
    return from_image(img, mask_volume_index=mask_volume_index)

def from_image(img, mask_volume_index=None):
    '''
    Parameters
    ----------
    img : nibabel SpatialImage
    
    Returns
    -------
    vg: VolGeom
        Volume geometry assocaited with img
    mask_volume_index: int or None (default)
        which volume in img to use as a voxel mask. None means no mask. 
    
    '''

    if not isinstance(img, nb.spatialimages.SpatialImage):
        raise TypeError("Image is not a spatial image: %r" % img)

    if mask_volume_index is None:
        mask = None
    else:
        data = img.get_data()
        if len(data.shape) == 3:
            mask = data
        else:
            idx = mask_volume_index if type(mask_volume_index) is int else 0
            mask = img.get_data()[:, :, :, idx]

    return VolGeom(shape=img.shape, affine=img.get_affine(), mask=mask)

