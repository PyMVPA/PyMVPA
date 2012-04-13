'''
Created on Mar 19, 2012

@author: nick
'''

import cPickle as pickle
import utils
import numpy as np
import volgeom

class Surface_Disc(object):
    '''quick and dirty attempt to replicate mvpa2 Sphere functionality
    but now for the surface.
    
    Currently it only accepts voxel selection results as stored by 
    pysurfing.sparse_volmasks.py; in the future we want to do this
    on the fly.
    
    '''
    def __init__(self, voxselfn):
        with open(voxselfn) as f:
            self._sel=pickle.load(f)
            self._keys=self._sel._n2roi.keys()
            
            # small refers to the indices 0..(n-1)  [for n nodes that have voxels associated with them)
            # big refers to a subset of 0..(m-1) if the original surface has m nodes
            # it should hold that n<=m 
            
            # for this initial implementation we just ignore this stuff
            # but it may become critical if we have surfaces with a lot of nodes
            # and not too many voxels
            self._s2b=dict(enumerate(self._keys)) # small to big
            self._b2s=dict((p,q) for q,p in enumerate(self._keys)) # big to small
            
    
    def __call__(self, coordinate):
        if not type(coordinate) is tuple:
            raise TypeError('Input should be a tuple')
        
        if len(coordinate)!=3:
            raise ValueError('Coordinate should have 3 values')
        
        c_ijk=np.asarray([[coordinate[0],coordinate[1],coordinate[2]]])
        
        vg=self._sel.volgeom
        if not vg.ijkinvol(c_ijk): # illegal value (too big, probably), return nothing
            return []
        
        c_lin=vg.ijk2lin(c_ijk)
        
        if not c_lin in self._keys: # no nodes associated
            return []
        
        assert len(c_lin)==1
        
        a_lin=self._sel.getroi(c_lin[0])
        a_ijk=vg.lin2ijk(a_lin)
        
        a_tuples=[(p[0],p[1],p[2]) for p in a_ijk]
        
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


if __name__ == '__main__':
    d=utils._get_fingerdata_dir()
    voxselfn=d+'glm/_voxsel1_.pickle'
    
    disc=Surface_Disc(voxselfn)