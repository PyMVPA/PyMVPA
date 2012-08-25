'''
Volume geometry to map between world and voxel coordinates.

Supports conversion between linear and sub indexing of voxels. The rationale is that
volumes use sub indexing that incorporate the spatial locations of voxels, but for 
voxel selection (and subsequent MVPA) we want to abstract from this spatial information 

Created on Feb 12, 2012

@author: nick
'''

import nibabel as ni, numpy as np, surf_fs_asc, utils

class VolGeom():
    '''
    Parameters
    ----------
    shape: tuple
        Number of items in each dimension.
        Typically the first three dimensions are spatial and the remaining ones temporal
    affine: numpy.ndarray
        4x4 affine transformation array that maps voxel to world coordinates
    '''
    def __init__(self, shape, affine):
        self._shape = shape
        self._affine = affine

    def as_pickable(self):
        '''
        Returns
        -------
        dict
            A dictionary that contains all information from this instance (and can be 
            saved using pickle)
        '''
        d = dict(shape=self.shape(), affine=self.affine())
        return d

    def _ijkmultfac(self):
        '''multiplication factors for ijk <--> linear indices conversion'''
        sh = self.shape
        return [sh[1] * sh[2], sh[2], 1]

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

        m = np.zeros((3,), dtype=int)
        fs = self._ijkmultfac()

        # make a 3x1 vector with multiplication factors
        for i, f in enumerate(fs):
            m[i] = f

        r = np.dot(ijk, m)
        return r

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

        lin = lin.copy() # we'll change lin, don't want to mess with input
        outsidemsk = np.logical_or(lin < 0, lin >= self.nvoxels)

        n = np.shape(lin)[0]
        fs = self._ijkmultfac()

        ijk = np.zeros((n, 3), dtype=int)
        for i, f in enumerate(fs):
            v = lin / f
            ijk[:, i] = v[:]
            lin -= v * f

        ijk[outsidemsk, :] = self.shape[:3]

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

        #ijk=np.array(ijkfloat,dtype=int)
        outsidemsk = np.logical_not(self.contains_ijk(ijk)) # invert means logical not


        # assign value of shape, which should give an out of bounds exception
        # if this value is actually used to index voxels in the volume
        sh = self.shape[:3]

        ijk[outsidemsk, :] = sh
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

        outsidemsk = np.invert(self.contains_ijk(ijk))
        xyz[outsidemsk, :] = np.NaN
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
        shape = self.shape

        return reduce(np.logical_and, [0 <= ijk[:, 0], ijk[:, 0] < shape[0],
                                      0 <= ijk[:, 1], ijk[:, 1] < shape[1],
                                      0 <= ijk[:, 2], ijk[:, 2] < shape[2]])

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

        nv = self.nvoxels
        return np.logical_and(0 <= lin, lin < nv)

    def empty_nifti_img(self, nt=1):
        sh = self.shape
        sh4d = (sh[0], sh[1], sh[2], nt)

        data = np.zeros(sh4d)
        img = ni.Nifti1Image(data, self.affine)
        return img

def from_nifti_file(fn):
    '''
    Parameters
    ----------
    fn : str
        Nifti filename
    
    Returns
    -------
    vg: VolGeom
        Volume geometry associated with 'fn'
    '''

    img = ni.load(fn)
    return from_image(img)

def from_image(img):
    '''
    Parameters
    ----------
    img : nibabel SpatialImage
    
    Returns
    -------
    vg: VolGeom
        Volume geometry assocaited with img
    '''

    if not isinstance(img, ni.spatialimages.SpatialImage):
        raise TypeError("Image is not a spatial image: %r" % img)

    return VolGeom(shape=img.get_shape(), affine=img.get_affine())

