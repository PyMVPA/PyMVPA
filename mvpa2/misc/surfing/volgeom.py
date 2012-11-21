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

Supports conversion between linear and sub indexing of voxels. 
The rationale is that volumes use sub indexing that incorporate the spatial 
locations of voxels, but for voxel selection (and subsequent MVPA) it is
often more appropriate to abstract from the temporal locations of voxels.

Created on Feb 12, 2012

@author: nick
'''

__docformat__ = 'restructuredtext'


import numpy as np

from mvpa2.base import warning, externals

if externals.exists('nibabel'):
    # nibabel is optional dependency here, those how would reach those points
    # without having nibabel should suffer
    import nibabel as nb

class VolGeom(object):
    '''Defines a mapping between sub and linear indices and world coordinate
    in volumatric fmri datasets'''

    def __init__(self, shape, affine, mask=None):
        '''
        Parameters
        ----------
        shape: tuple
            Number of values in each dimension.
            Typically the first three dimensions are spatial and the remaining ones
            temporal.
        affine: numpy.ndarray
            4x4 affine transformation array that maps voxel to world coordinates.
        mask: numpy.ndarray (default: None)
            voxel mask that indicates which voxels are included. Values of zero in 
            mask mean that a voxel is not included. If mask is None, then all 
            voxels are included.
        
        '''
        self._shape = shape
        self._affine = affine
        if not mask is None:
            if mask.size != self.nvoxels:
                raise ValueError("%d voxels, but mask has %d" %
                                 (self.nvoxels, mask.size))
            mask = np.reshape(mask != 0, (-1,))
        self._mask = mask

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (self.shape == other.shape and
                np.all(self.mask == other.mask) and
                np.all(self.affine == other.affine))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, prefixes=[]):
        prefixes_ = ['shape=(%s)' % ','.join(['%r' % i for i in self._shape]),
                     'affine=%r' % self._affine] + prefixes
        if not self._mask is None:
            prefixes_ += ['mask=%r' % self._mask]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(prefixes_))

    def __str__(self):
        sh = self.shape[:3]
        s = '%s, %s = %d voxels' % (self.__class__.name,
                             '%d x %d x %d' % sh, self.nvoxels)
        if not self.mask is None:
            s += ', %d voxels survive the mask' % self.nvoxels_mask


    def as_pickable(self):
        '''
        Returns a pickable instance.
        
        Returns
        -------
        dict
            A dictionary that contains all information from this instance 
            (and can be saved using pickle).
        '''
        d = dict(shape=self.shape, affine=self.affine, mask=self.mask)
        return d

    def __reduce__(self):
        return (self.__class__, (self._shape, self._affine, self._mask))

    @property
    def mask(self):
        '''
        Returns the mask.
        
        Returns
        -------
        mask: np.ndarray
            boolean vector indicating which voxels are included.
            If no mask is associated with this instance then mask is None.
        '''
        if self._mask is None:
            return None

        m = self._mask.view()
        m.flags.writeable = False
        return m

    @property
    def linear_mask(self):
        '''Returns the mask as a vector
        
        Returns
        -------
        mask: np.ndarray (vector with P values)
            boolean vector indicating which voxels are included.
            If no mask is associated with this instance then mask is None.
        '''
        if self._mask is None:
            return None

        return np.reshape(self.mask, (self.nvoxels,))

    def _ijkmultfac(self):
        '''multiplication factors for ijk <--> linear indices conversion'''
        sh = self.shape
        return [sh[1] * sh[2], sh[2], 1]

    def _contains_ijk_unmasked(self, ijk):
        '''helper function to see if ijk indices are in the volume'''
        shape = self.shape

        m = reduce(np.logical_and, [0 <= ijk[:, 0], ijk[:, 0] < shape[0],
                                   0 <= ijk[:, 1], ijk[:, 1] < shape[1],
                                   0 <= ijk[:, 2], ijk[:, 2] < shape[2]])
        return m

    def _outside_vol(self, ijk, lin):
        '''helper function to see if ijk and lin indices are in the volume.
        It is assumed that these indices are matched (i.e. ijk[i,:] and lin[i]
        refer to the same voxel, for all i).
        
        The rationale for providing both is that ijk is necessary to determine
        whether a voxel is within the volume, while lin is necessary for 
        considering which voxels are in the mask'''
        invol = self._contains_ijk_unmasked(ijk)
        invol[np.logical_or(lin < 0, lin >= self.nvoxels)] = np.False_
        invol[np.isnan(lin)] = np.False_

        if not self.mask is None:
            invol[invol] = np.logical_and(invol[invol], self.mask[lin[invol]])

        return np.logical_not(invol)

    def _ijk2lin_unmasked(self, ijk):
        '''helper function to convert sub to linear indices.'''
        m = np.zeros((3,), dtype=int)
        fs = self._ijkmultfac()

        # make a 3x1 vector with multiplication factors
        for i, f in enumerate(fs):
            m[i] = f

        lin = np.dot(ijk, m)
        return lin

    def _lin2ijk_unmasked(self, lin):
        '''Converts sub to linear voxel indices
        
        Parameters
        ----------
            Px1 array with linear voxel indices
            
        Returns
        -------
        ijk: numpy.ndarray
            Px3 array with sub voxel indices
        '''

        if not isinstance(lin, np.ndarray):
            lin = np.asarray(lin, dtype=np.int)
        else:
            lin = np.copy(lin)
        lin = lin.ravel()

        n = np.shape(lin)[0]
        fs = self._ijkmultfac()

        ijk = np.zeros((n, 3), dtype=int)
        for i, f in enumerate(fs):
            v = lin / f
            ijk[:, i] = v[:]
            lin -= v * f

        return ijk

    def ijk2triples(self, ijk):
        '''Converts sub indices to a list of triples
        
        Parameters
        ----------
        ijk: np.ndarray (Px3)
            sub indices
            
        Returns
        -------
        triples: list with P triples
            the indices from ijk, so that triples[i][j]==ijk[i,j]
        '''

        return map(tuple, ijk)

    def triples2ijk(self, tuples):
        '''Converts triples to sub indices
        
        Parameters
        ----------
        triples: list with P triples
        
        Returns
        -------
        ijk: np.ndarray(Px3)
            an array from triples, so that ijk[i,j]==triples[i][j]
        '''

        return np.asarray(tuples)

    def ijk2lin(self, ijk):
        '''Converts sub to linear voxel indices.
        
        Parameters
        ----------
        ijk: numpy.ndarray
            Px3 array with sub voxel indices.
        
        Returns
        -------
        lin: Px1 array with linear voxel indices.
            If ijk[i,:] is outside the volume, then lin[i]==self.nvoxels.
        '''

        lin = self._ijk2lin_unmasked(ijk)
        lin[self._outside_vol(ijk, lin)] = self.nvoxels

        return lin

    def lin2ijk(self, lin):
        '''Converts sub to linear voxel indices.
        
        Parameters
        ----------
            Px1 array with linear voxel indices.
            
        Returns
        -------
        ijk: numpy.ndarray
            Px3 array with sub voxel indices.
            If lin[i] is outside the volume, then ijk[i,:]==self.shape.
        '''

        ijk = self._lin2ijk_unmasked(lin)
        ijk[self._outside_vol(ijk, lin), :] = self.shape[:3]

        return ijk

    @property
    def affine(self):
        '''Returns the affine transformation matrix.
        
        Returns
        -------
        affine : numpy.ndarray
            4x4 array that maps voxel to world coordinates.
        '''

        a = self._affine.view()
        a.flags.writeable = False
        return a


    def xyz2ijk(self, xyz):
        '''Maps world coordinates to sub voxel indices.
        
        Parameters
        ----------
        xyz : numpy.ndarray (float)
            Px3 array with world coordinates.
        
        Returns
        -------
        ijk: numpy.ndarray (int)
            Px3 array with sub voxel indices.
            If xyz[i,:] is outside the volume, then ijk[i,:]==self.shape
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
        '''Maps sub voxel indices to world coordinates.
        
        Parameters
        ----------
        ijk: numpy.ndarray (int)
            Px3 array with sub voxel indices.
        
        Returns
        -------
        xyz : numpy.ndarray (float)
            Px3 array with world coordinates.
            If ijk[i,:] is outside the volume, then xyz[i,:] is NaN.
        '''

        m = self.affine
        ijkfloat = np.array(ijk, dtype=float)
        xyz = self.apply_affine3(m, ijkfloat)

        lin = self._ijk2lin_unmasked(ijk)
        self._outside_vol(ijk, lin)

        xyz[self._outside_vol(ijk, lin), :] = np.NaN
        return xyz


    def xyz2lin(self, xyz):
        '''Maps world coordinates to linear voxel indices.
        
        Parameters
        ----------
        xyz : numpy.ndarray (float)
            Px3 array with world coordinates
        
        Returns
        -------
        ijk: numpy.ndarray (int)
            Px1 array with linear indices.
            If xyz[i,:] is outside the volume, then lin[i]==self.nvoxels.
        '''
        return self.ijk2lin(self.xyz2ijk(xyz))

    def lin2xyz(self, lin):
        '''Maps linear voxel indices to world coordinates.
        
        Parameters
        ----------
        ijk: numpy.ndarray (int)
            Px3 array with linear voxel indices.
        
        Returns
        -------
        xyz : np.ndarray (float)
            Px1 array with world coordinates.
            If lin[i] is outside the volume, then xyz[i,:] is NaN.
        '''

        return self.ijk2xyz(self.lin2ijk(lin))

    def apply_affine3(self, mat, v):
        '''Applies an affine transformation matrix.
        
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
        Returns the number of time points.
        
        Returns
        -------
        int
            Number of time points
        '''
        return np.prod(self.shape[3:])
    @property
    def nvoxels(self):
        '''
        Returns the number of voxels.
        
        Returns
        -------
        int
            Number of spatial points (i.e. number of voxels)
        '''
        return np.prod(self.shape[:3])

    @property
    def shape(self):
        '''
        Returns the shape.
        
        Returns
        -------
        tuple: int
            Number of values in each dimension
        '''

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
        Returns whether a set of sub voxel indices are contained
        within this instance.
        
        Parameters
        ----------
        ijk : numpy.ndarray
            Px3 array with sub voxel indices
        
        Returns
        -------
        numpy.ndarray (boolean)
            P boolean values indicating which voxels are within the volume.
        '''
        lin = self._ijk2lin_unmasked(ijk)

        return np.logical_not(self._outside_vol(ijk, lin))


    def contains_lin(self, lin):
        '''
        Returns whether a set of linear voxel indices are contained
        within this instance.
        
        Parameters
        ----------
        lin : numpy.ndarray
            Px1 array with linear voxel indices.
        
        Returns
        -------
        numpy.ndarray (boolean)
            P boolean values indicating which voxels are within the volume.
        '''

        ijk = self._lin2ijk_unmasked(lin)

        return np.logical_not(self._outside_vol(ijk, lin))

    def empty_nifti_img(self, nt=None):
        sh = self.shape

        if not nt is None:
            sh = (sh[0], sh[1], sh[2], nt)

        data = np.zeros(sh)
        img = nb.Nifti1Image(data, self.affine)
        return img

    def masked_nifti_img(self, nt=None, neighborhood_func=None):
        '''Provides a masked nifti image
        
        Parameters
        ----------
        nt: int or None
            Number of timepoints (or samples). Each feature has the
            same value (1 if in the mask, 0 otherwise) for each
            sample. If nt is None, then the output is 3D; otherwise
            it is 4D with 'nt' values in the last dimension.
        neighborhood_func: callable or None
            A neighborhood function (like Sphere(..) that can map 
            a single coordinate (represented as a triple of 3D indices
            to a list of triples that define the neighboorhood of that
            coordinate. For example, Sphere(3) can be used to dilate the
            original mask by 3 voxels.
            
        Returns
        -------
        msk: Nifti1image
            a nifti image with values 1. for values inside the mask
            and values of 0 elsewhere. If the instance has no mask,
            then all values are 1. 
        '''
        data_lin = np.zeros((self.nvoxels, nt or 1), dtype=np.float32)
        if self.mask is None:
            data_lin[:, :] = 1
        else:
            data_lin[self.mask, :] = 1

        sh = self.shape
        if not nt is None:
            sh = (sh[0], sh[1], sh[2], nt)

        data = np.reshape(data_lin, sh)

        if not neighborhood_func is None and \
                    self.nvoxels_mask != self.nvoxels:

            for ijk_center in zip(*np.nonzero(data)):
                ijks_around = neighborhood_func(ijk_center)
                for ijk_around in ijks_around:
                    data[ijk_around] = 1

        img = nb.Nifti1Image(data, self.affine)
        return img


def from_any(s, mask_volume=None):
    """Constructs a VolGeom instance from any reasonable type of input.
    
    Parameters
    ----------
    s : str or VolGeom or nibabel SpatialImage-like or 
                mvpa2.datasets.base.Dataset-like with nifti-image header.
        Input to use to construct the VolGeom instance. If s is a string,
        then it is assumed to refer to the file name of a NIFTI image.
    mask_volume: boolean or int or None (default: False)
        If an int is provided, then the mask-volume-th volume in s
        is used as a voxel mask. True is equivalent to 0. If None or
        False are provided, no mask is applied.
        In the special case that s is fmri_dataset-like and mask_volume
        is a string (such as 'voxel_indices'), then it is assumed
        that s.fa[mask_volume] is a list with voxel coordinates to be
        included.
    
    Returns
    -------
    vg: VolGeom
        Volume geometry associated with s.
    """
    if s is None or isinstance(s, VolGeom):
        return s

    if isinstance(s, basestring):
        # try to find a function to load the data
        load_function = None

        if s.endswith('.nii') or s.endswith('.nii.gz'):
            load_function = nb.load
        elif s.endswith('.h5py'):
            if externals.exists('h5py'):
                from mvpa2.base.hdf5 import h5load
            else:
                raise ValueError("Cannot load h5py file - no externals")

        if load_function:
            # do a recursive call
            return from_any(load_function(s), mask_volume=mask_volume)

        raise ValueError("Unrecognized extension for file %s" % s)

    if mask_volume is True:
        # assign a specific index -- the very first volume
        mask_volume = 0
    elif mask_volume is False:
        mask_volume = None

    try:
        # see if s behaves like a spatial image (nifti image)
        shape = s.shape
        affine = s.get_affine()

        if isinstance(mask_volume, int):
            data = s.get_data()
            ndim = len(data.shape)
            if ndim <= 3:
                mask = data
                if mask_volume > 0:
                    warning("There is no 4th dimension (t) to select "
                            "the %d-th volume." % (mask_volume,))
            else:
                mask = data[:, :, :, mask_volume]
        else:
            mask = None
    except:
        try:
            # see if s behaves like a Dataset with image header
            hdr = s.a.imghdr

            shape = hdr.get_data_shape()
            affine = hdr.get_best_affine()

            mask = None
            if isinstance(mask_volume, int):
                mask = np.asarray(s.samples[mask_volume, :])
            else:
                mask_volume_indices = None
                if mask_volume is None and (hasattr(s, 'fa') and
                                           hasattr(s.fa, 'voxel_indices')):
                    mask_volume_indices = s.fa['voxel_indices']
                elif isinstance(mask_volume, basestring):
                    if not mask_volume in s.fa:
                        raise ValueError('Key not found in s.fa: %r' % mask_volume)
                    mask_volume_indices = s.fa[mask_volume]

                if mask_volume_indices:
                    sh = shape[:3]
                    mask = np.zeros(sh)

                    for idx in mask_volume_indices.value:
                        mask[tuple(idx)] = 1
        except:
            #no idea what type of beast this is.
            raise ValueError('Unrecognized input %r - not a VolGeom, '
                             '(filename of) Nifti image, or (mri-)Dataset' % s)

    return VolGeom(shape=shape, affine=affine, mask=mask)


def distance(p, q, r=2):
    '''Returns the distances between vectors in two arrays
    
    Parameters
    ----------
    p: np.ndarray (PxM)
        first array
    q: np.ndarray (QxM)
        second array
    nrm: float (default: 2)
        Norm used for distance computation. By default Euclidian distances
        are computed.
        
    Returns
    -------
    pq: np.ndarray (PxQ)
        Distance between p[j] and q[j] is in pq[i,j]
    '''

    if p.shape[1] != q.shape[1]:
        raise ValueError("Shape mismatch")

    m, n = len(p), len(q)
    ds = np.zeros((m, n), dtype=p.dtype)

    def dist_func(a, b, r):
        delta = a - b
        if np.isinf(r):
            return np.max(np.abs(delta), 1)
        else:
            return np.sum(delta ** r, 1) ** (1. / r)

    for i, pi in enumerate(p):
        ds[i, :] = dist_func(pi, q, r)

    return ds
