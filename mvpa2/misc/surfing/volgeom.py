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
    def __init__(self,shape,affine):
        self._shape=shape
        self._affine=affine
        
    def as_pickable(self):
        '''
        Returns
        -------
        dict
            A dictionary that contains all information from this instance (and can be 
            saved using pickle)
        '''
        d=dict(shape=self.shape(),affine=self.affine())
        return d
        
    def _ijkmultfac(self):
        '''multiplication factors for ijk <--> linear indices conversion'''
        sh=self.shape()
        return [sh[1]*sh[2],sh[2],1]      
        
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
        
        m=np.zeros((3,),dtype=int)
        fs=self._ijkmultfac()
        
        # make a 3x1 vector with multiplication factors
        for i,f in enumerate(fs):
            m[i]=f
        
        r=np.dot(ijk,m)
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
        
        lin=lin.copy() # we'll change lin, don't want to mess with input
        outsidemsk=np.logical_or(lin<0,lin>=self.nv())
        
        n=np.shape(lin)[0]
        fs=self._ijkmultfac()
        
        ijk=np.zeros((n,3),dtype=int)
        for i,f in enumerate(fs):
            v=lin/f
            ijk[:,i]=v[:]
            lin-=v*f
        
        ijk[outsidemsk,:]=self.shape()
        
        return ijk
    
    def affine(self):
        '''Returns the affine transformation matrix
        
        Returns
        -------
        affine : numpy.ndarray
            4x4 array that maps voxel to world coorinates
        '''
        
        return self._affine
        
        
    def xyz2ijk(self,xyz):
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
        m=self.affine()
        minv=np.linalg.inv(m)
        
        ijkfloat=self.apply_affine3(minv,xyz)
        
        # add .5 so that positions are rounded instead of floored CHECKME
        ijk=np.array(ijkfloat+.5,dtype=int)
        #ijk=np.array(ijkfloat,dtype=int)
        outsidemsk=np.logical_not(self.ijkinvol(ijk)) # invert means logical not
        
        
        # assign value of shape, which should give an out of bounds exception
        # if this value is actually used to index voxels in the volume
        sh=self.shape()
        ijk[outsidemsk,:]=sh
        return ijk
    
    def ijk2xyz(self,ijk):
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
        
        m=self.affine()
        ijkfloat=np.array(ijk,dtype=float)
        xyz=self.apply_affine3(m, ijkfloat)
        
        outsidemsk=np.invert(self.ijkinvol(ijk))
        xyz[outsidemsk,:]=np.NaN
        return xyz
        
    
    def xyz2lin(self,xyz):
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
    
    def lin2xyz(self,lin):
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
    
    def apply_affine3(self,mat,v):
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
        
        
        r=mat[:3,:3]
        t=mat[:3,3].transpose()
        
        return np.dot(v,r)+t
        
    def nt(self):
        '''
        Returns
        -------
        int
            Number of time points
        '''
        return np.prod(self.shape()[3:])
        
    def nv(self):
        '''
        Returns
        -------
        int
            Number of spatial points (i.e. number of voxels)
        '''
        return np.prod(self.shape()[:3])
    
    def shape(self):
        '''
        Returns
        -------
        tuple: int
            Number of values in each dimension'''
        return self._shape
    
    def ijkinvol(self,ijk):
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
        shape=self.shape()
        
        return reduce(np.logical_and,[0<=ijk[:,0],ijk[:,0]<shape[0],
                                      0<=ijk[:,1],ijk[:,1]<shape[1],
                                      0<=ijk[:,2],ijk[:,2]<shape[2]])
        
    def lininvol(self,lin):
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
        
        nv=self.nv()
        return np.logical_and(0<=lin,lin<nv)
        
    def _testlindices(self):
        '''just for testing'''
        data=self._img.get_data()
        shape=np.shape(data)
        nt=np.prod(shape[3:])
        print nt
        datars=np.reshape(data, (-1,nt))
        
        c=np.array([[10,10,10]])
        r=range(-2,3)
        
        datars[:]=0
        print np.shape(datars)
        for i in r:
            for j in r:
                for k in r:
                    ijk=np.array([[i,j,k]])+c
                    linindex=self.ijk2lin(ijk)
                    ijk2=self.lin2ijk(linindex)
                    if (ijk2!=ijk).any():
                        raise Exception("Unexpected")
                    
                    datars[linindex,0]=i
                    datars[linindex,1]=j
                    datars[linindex,2]=k
                    datars[linindex,3]=linindex
        
        databack=np.reshape(datars,np.shape(data))
        
        img=ni.Nifti1Image(databack,self._img.get_affine())    
        return img
        
        
    def _getmaskedimage(self,xyz):
        '''just for testing'''
        data=self._img.get_data();
        datars=np.reshape(data,(-1,self.nt()))
        
        ijk=self.xyz2ijk(xyz)
        
        nrows=np.shape(ijk)[0]
        sh=self.shape()
        
        msk=self.ijkinvol(ijk)
        
        print np.shape(datars)
        print np.shape(msk)
        
        data[:]=0
        data[ijk[msk]]=[1,2,3]
        

        img=ni.Nifti1Image(np.reshape(data,self.shape()),self._img.get_affine())
        return img

def from_nifti_filename(fn):
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
    
    img=ni.load(fn)
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
    
    return VolGeom(shape=img.get_shape(),affine=img.get_affine())



if __name__ == '__main__':
    d='/Users/nick/Downloads/fingerdata-0.2/glm/'
    fn=d+"rall_vol00.nii"
    surffn=d+"ico100_lh.smoothwm_al.asc"
    Surface=surf_fs_asc.read(surffn)
    vg=from_nifti_filename(fn)
    
    data=np.zeros(vg.shape())
    
    xyz=Surface.v()
    ijk=vg.xyz2ijk(xyz)
    
    nv=ijk.shape[0]
    invol=vg.ijkinvol(ijk)
    
    for kk in [43247]: #xrange(nv):
        i,j,k=ijk[kk,0],ijk[kk,1],ijk[kk,2]
        sh=data.shape
        
        if 0<=i and i<sh[0] and 0<=j and j<sh[1] and 0<=k and k<sh[2]:
            data[i,j,k]+=1
    
    fnout=d+"__q4.nii"
    img=ni.Nifti1Image(data,vg.affine())
    img.to_filename(fnout)
        
    
''' *** ONLY TESTING FROM HERE *** '''       
    
if __name__ == '__main__':
    d='%s/glm/' % utils._get_fingerdata_dir()
    fn=d+"rall_vol00.nii"
    surffn=d+"ico100_lh.smoothwm_al.asc"
    Surface=surf_fs_asc.read(surffn)
    nodeidx=43247
    xyz=np.reshape(Surface.v()[nodeidx,:],(-1,3))
    print "xyz %r" % xyz
    
    linidx=109543
    i,j,k=48,31,21
    ijk=np.asarray([[i,j,k]],dtype=int)
    
    
    vg=from_nifti_filename(fn)
    linX=vg.ijk2lin(ijk)
    print "lin %r %r (or %r) " % (linidx, linX, vg.nv()-linidx)
     
    linY=vg.xyz2lin(xyz)
    ijkY=vg.xyz2ijk(xyz)
    print "xyz2 %r %r (expected ijk %r)" % (linY, ijkY, ijk)
    
    m=vg.affine()
    
    
    tf=vg.apply_affine3(m,ijk)
    print "orig/expected %r / %r" % (ijk,tf)
    
    mi=np.linalg.inv(m)
    tfa=vg.apply_affine3(mi, xyz)
    
    linarray=np.asarray([linidx])
    print linarray
    myijk=vg.lin2ijk(linarray)
    
    print "orig/expected %r (%r) / %r" % (myijk,ijk,tfa)
    
    print vg.ijkinvol(ijk)
    
    

    
    
    
if __name__ == '__mainX__':    
    d='/Users/nick/Downloads/fingerdata-0.2/glm/'
    fn=d+"rall_vol00.nii"
    fn2=d+"rall.nii"
    fn3=d+"__anat_hires.nii"
    fn3=d+"anat_al.nii"
    fn3=d+"__anat10.nii"
    fnout=d+"__test9b.nii"
    #surffn=d+"../ref/ico100_lh.pial_al.asc"
    surffn=d+"../ref/ico100_lh.smoothwm_al.asc"
    v=from_nifti_filename(fn)
    print "started"
    
    
    print v
    print v.shape()
    ijk=np.array([[1,7,20],[2,7,20],[1,8,20],[1,7,21]])
    
    msk=np.empty((4,3),dtype=bool)
    m2=np.empty((4,),dtype=bool)
    
    
    #img=v._testlindices()
    s=surf_fs_asc.read(surffn)
    
    ijk=v.xyz2ijk(s.v())
    
    invol=v.ijkinvol(ijk)
    print invol
    print invol.shape
    
    lin=v.ijk2lin(ijk)
    invol2=v.lininvol(lin)
    print invol2.shape
    

    nrows=np.shape(ijk)[0]
    sh=v.shape()
    
    msk=v.ijkinvol(ijk)
    data=v._img.get_data()
    xyz=s.v()
    lin=v.xyz2lin(xyz)
    rs=np.reshape(data,(-1,v.nt()))
    rs[:]=0
    rs[lin[msk]]=1
    data=np.reshape(rs,v.shape())
    img=ni.Nifti1Image(data,v.affine())
    
    #img=v._getmaskedimage(s.v())
    img.to_filename(fnout)
    print fnout
    '''
    
    
    
    ijk=v.xyz2ijk(m)
    xyz=v.ijk2xyz(ijk)
    
    s=surf_fs_asc.read(surffn)
    _img=v.getmaskedimage(s.v())
    
    fnout=d+"__vol1.nii"
    _img.to_filename(fnout)
    
    lin=v.ijk2lin(ijk)
    ijk2=v.lin2ijk(lin)
    
    print ijk
    print ijk2
    
    msk=v.ijkinvol(v.xyz2ijk(s._v))
    '''
    