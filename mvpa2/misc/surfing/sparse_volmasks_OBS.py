'''
Implementation for storing many sparse volume masks.
It is used to store voxel selection results that can be used for information mapping

TODO: integrate with pyMVPA, find acceptable data structure

Created on Feb 21, 2012

@author: nick
'''
import nibabel as ni, numpy as np
import collections
import volgeom
import cPickle as pickle 
import utils

_VOXIDXSLABEL="voxels" # name of key for linaer voxel indices
_VOXDISTLABEL="distances" 
_INTTYPE=np.int32 # data type to store integers
_FLOATTYPE=np.float32 # data type to store floats

class SparseVolMask():
    '''
    Initial implementation to store various sparse volume masks,
    used to store voxel selection results.
    
    Future plans is to change this so that it becomes a type (subclass?) of Dataset
    
    Parameters
    ----------
    volgeom: volgeom.VolGeom
        Volume geometry associated with 
    meta: dict
        Optional extra information about the volume
    
    '''
      
    def __init__(self,volgeom=None,meta=None):
        self.setvolgeom(volgeom)
        self._n2roi=collections.defaultdict(dict)
        if meta is None:
            self.meta=dict()
            
    def _to_proper_datatype(self,xs):
        '''helper function to ensure that data is stored in proper ints and floats'''
        if type(xs) is list:
            return map(self._to_proper_datatype,xs)
            
        if not isinstance(xs,np.ndarray):
            xs=np.asarray(xs)
        
        tp=xs.dtype.type
        if issubclass(tp,np.integer):
            return np.asarray(xs,dtype=_INTTYPE)
        elif issubclass(tp,np.float):
            return np.asarray(xs,dtype=_FLOATTYPE)
        else:
            raise TypeError("Unrecognized type %r" % tp)
        
    def getvolgeom(self):
        '''Sets the volume geometry'''
        return self.volgeom
    
    def setvolgeom(self,vg):
        '''Returns the volume geometry'''
        if vg is None:
            return
        if not isinstance(vg, volgeom.VolGeom):
            raise TypeError("Illegal volgeom %r" % vg)
        self.volgeom=vg
    
    def _checkidxs(self,idxs):
        '''Not really used now...'''
        vg=self.volgeom
        if vg is None:
            raise ValueError("no volgeom defined")
        
    def getroilabels(self):
        '''Get the labels of the ROIs. If used for surface-based voxel 
        selection, then it returns the node indices
        
        Returns
        -------
        roilabels: list
            Labels of masks (i.e. ROIs) 
        '''
        
        return self._n2roi.keys() 
    
    def addroi(self,roilabel,idxs):
        '''Add a region of interest (i.e. a sparse mask)
        
        Parameters
        ----------
        roilabel: 
            Usually the index of the node
        idxs: list-like
            Linear voxel indices to be associated with roilabel
        '''
        
        self._checkidxs(idxs)
        self.addroimeta(roilabel,{_VOXIDXSLABEL:idxs})
        
    def addroi_fromdict(self,roilabel,dct):
        '''Add a region of interest (i.e. a sparse mask) from a dictionary
        
        Parameters
        ----------
        roilabel: 
            Usually the index of the node
        dct: dict
            Dictionary with voxel indices and other properties associated
            with the sparse mask. It should contain a key with value
            equal to sparse_volmasks._VOXIDXSLABEL, and the associated
            value should be a list with linear indices corresponding
            to the voxels in the mask
        '''
        
        
        if not dct:
            return
        
        if not _VOXIDXSLABEL in dct:
            raise ValueError("No label %s in %r" % (_VOXIDXSLABEL, dct))
        
        for k,v in dct.iteritems():
            self._n2roi[roilabel][k]=self._to_proper_datatype(v)
    
    def getroi(self,roilabel):
        '''
        Get linear voxel indices for a single mask
        
        
        ''' 
        return self.getroimeta(roilabel, _VOXIDXSLABEL)
    
    def getroimeta(self,roilabel,metalabel):
        return self._n2roi[roilabel][metalabel]
        
    def getbinaryroimask(self,roilabel):
        linidxs=self.getroi(roilabel)
        vg=self.volgeom
        nvox=vg.nv()
        
        msk=np.zeros(shape=(nvox,),dtype=np.bool_)
        msk[linidxs]=np.True_
        return msk


class SurfaceDisc():
    def __init__(self,sel):
        self._sel=sel
        self._keys=self._sel._n2roi.keys()
        
        print "%d keys" % len(self._keys)
        
    def __call__(self, coordinate):
        c_ijk=np.asanyarray(coordinate)[np.newaxis]

        if not c_ijk.shape == (1, 3):
            raise ValueError('Coordinate should be an length-3 iterable')

        vg=self._sel.volgeom
        if not vg.ijkinvol(c_ijk): # illegal value (too big, probably), return nothing
            return tuple()
        
        c_lin=vg.ijk2lin(c_ijk)
        
        if not c_lin in self._keys: # no nodes associated
            return tuple()
        
        assert len(c_lin)==1
        
        a_lin=self._sel.getroi(c_lin[0])
        a_ijk=vg.lin2ijk(a_lin)
        
        a_tuples = [tuple(p) for p in a_ijk]
        
        return a_tuples
    
    def node_indices(self):
        '''Completely untested, but the following may work:
        
        assume ds is a dataset (S samples x F features matrix), and ds_t is
        the transpose version (as an np.ndarray with size FxS).
        
        (ds might be obtained by a proper slicing operation along the features,
        maybe something like ds[:,node_indices()]) 
        
        then:
            surfdset=dict(data=ds_t,node_indices=disc.node_indices)
            afni_niml_dset.write('myfile.niml.dset',surfdset)
            
        might work (fingers crossed)
        '''
        
        keys=self._keys
        return np.asarray(keys)      
        
    
    

def file2disc(fn):
    return SurfaceDisc(from_file(fn))
            
        
def to_file(fn,sel):
    if not hasattr(sel,'_n2roi'):
        raise ValueError("Not an sparse volmask")
    
    with open(fn,'w') as f:
        pickle.dump(sel, f, protocol=pickle.HIGHEST_PROTOCOL)

def _say_hello():
    print "HW"

def from_file(fn):
    
    print "Loading from ", fn
    with open(fn) as f:
        r=pickle.load(f)
    
    if not hasattr(r,'_n2roi'):
        raise ValueError("Not an sparse volmask")
    
    return r
        

if __name__ == '__main__':
    d='/Users/nick/Downloads/fingerdata-0.2/glm/'
    epifn=d+"rall_vol00.nii"
    
    pialfn=d+"ico100_lh.pial_al.asc"
    whitefn=d+"ico100_lh.smoothwm_al.asc"
    
    fninprefix=d+"_voxsel1_"
    
    selfn=fninprefix+".pickle"
    
    sel=from_file(selfn)
    
    print sel
    
    