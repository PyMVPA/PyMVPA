import cPickle as pickle
import volgeom
from collections import Mapping

import nibabel as ni, numpy as np
import collections
import volgeom
import cPickle as pickle 
import utils

from mvpa2.misc.neighborhood import IndexQueryEngine
from mvpa2.measures.searchlight import Searchlight

class SparseAttributes(object):
    def __init__(self,sa_labels):
        self._sa_labels=list(sa_labels)
        self.sa=dict()
        self.a=dict()
        
    def set(self,roi_label,roi_attrs):
        if not roi_attrs:
            roi_attrs=None
        else:
            if not type(roi_attrs) is dict:
                raise TypeError("roi attributes should be a dict, but is %r:\n%r" %
                                (type(roi_attrs),roi_attrs))
            
            # if sa_labels is set, check it has all the keys
            if set(self._sa_labels)!=set(roi_attrs):
                raise ValueError("Key set mismatch: %r != %r" % 
                                 (self._sa_labels,roi_attrs))
        
        self.sa[roi_label]=roi_attrs
        
        if not roi_attrs is None:
            self._sa_nonempty_keys.append(roi_label)
        
    def add(self,roi_label,roi_attrs):
        if roi_label in self.sa.keys():
            raise ValueError("name clash: key %s already present" % roi_label)
        self.set(roi_label, roi_attrs)
    
    def sa_labels(self):
        return list(self._sa_labels)
    
    def keys(self):
        return [k for k in self.all_keys() if not k is None]
    
    def all_keys(self):
        return self.sa.keys()
    
    def get(self, roi_label, sa_label):
        roiattr=self.sa[roi_label]
        return roiattr[sa_label] if roiattr else None
    
    def get_attr_mapping(self, roi_attr):
        '''Provides a dict-like object with lookup for a 
        single ROI attribute'''
        
        if not roi_attr in self.sa_labels():
            raise KeyError("attribute %r not in map" % roi_attr)
        
        class AttrMapping(Mapping):
            def __init__(self,cls,roi_attr):
                self._roi_attr=roi_attr
                self._cls=cls
                
            def __getitem__(self,key):
                return self._cls.get(key,self._roi_attr)
            
            def __len__(self):
                return len(self.__keys__())
            
            def __keys__(self):
                return list(self._cls.keys())
            
            def __iter__(self):
                return iter(self.__keys__())
            
        return AttrMapping(self,roi_attr) 
    
    def __repr__(self):
        return ("SparseAttributes with %i entries, %i labels (%r)\nGeneral attributes: %r" % 
                (len(self.sa), len(self._sa_labels), self._sa_labels, self.a.keys()))
        
    
    
        

class SparseVolumeAttributes(SparseAttributes):
    def __init__(self,sa_labels,volgeom):
        super(self.__class__,self).__init__(sa_labels)
        self.a['volgeom']=volgeom
        
    def get_neighborhood(self,voxel_ids_label='lin_vox_idxs'):
        return SparseNeighborhood(self,voxel_ids_label)
        

class SparseNeighborhood():
    def __init__(self,attr,voxel_ids_label='lin_vox_idxs'):
        if not voxel_ids_label in attr.sa_labels():
            raise ValueError("%r is not a valid key in %r" % (voxel_ids_label, attr))
        
        self._attr=attr
        self._voxel_ids_label=voxel_ids_label
        
    def __call__(self, coordinate):
        c_ijk=np.asanyarray(coordinate)[np.newaxis]

        if not c_ijk.shape == (1, 3):
            raise ValueError('Coordinate should be an length-3 iterable')

        vg=self._attr.a['volgeom']
        if not vg.ijkinvol(c_ijk): # illegal value (too big, probably), return nothing
            return tuple()
        
        c_lin=vg.ijk2lin(c_ijk)
        
        if not c_lin in self._attr.keys(): # no nodes associated
            return tuple()
        
        if len(c_lin)!=1:
            raise ValueError("No unique value for c_lin (found %d)"
                             "; something went wrong" % len(c_lin))
        
        a_lin=self._attr.get(c_lin[0],self._voxel_ids_label)
        if a_lin is None:
            return tuple()
        
        a_ijk=vg.lin2ijk(a_lin)
        
        a_tuples = [tuple(p) for p in a_ijk]
        
        return a_tuples
    
def searchlight(datameasure, neighborhood, center_ids=None,
                       space='voxel_indices', **kwargs):
    """Creates a `Searchlight` to run a scalar `Measure` on
    all possible spheres of a certain size within a dataset.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.

    Parameters
    ----------
    datameasure : callable
      Any object that takes a :class:`~mvpa2.datasets.base.Dataset`
      and returns some measure when called.
    radius : int
      All features within this radius around the center will be part
      of a sphere. Radius is in grid-indices, i.e. ``1`` corresponds
      to all immediate neighbors, regardless of the physical distance.
    center_ids : list of int
      List of feature ids (not coordinates) the shall serve as sphere
      centers. Alternatively, this can be the name of a feature attribute
      of the input dataset, whose non-zero values determine the feature
      ids.  By default all features will be used (it is passed as ``roi_ids``
      argument of Searchlight).
    space : str
      Name of a feature attribute of the input dataset that defines the spatial
      coordinates of all features.
    **kwargs
      In addition this class supports all keyword arguments of its
      base-class :class:`~mvpa2.measures.base.Measure`.

    Notes
    -----
    If `Searchlight` is used as `SensitivityAnalyzer` one has to make
    sure that the specified scalar `Measure` returns large
    (absolute) values for high sensitivities and small (absolute) values
    for low sensitivities. Especially when using error functions usually
    low values imply high performance and therefore high sensitivity.
    This would in turn result in sensitivity maps that have low
    (absolute) values indicating high sensitivities and this conflicts
    with the intended behavior of a `SensitivityAnalyzer`.
    """
    # build a matching query engine from the arguments
    kwa = {space: neighborhood}
    qe = IndexQueryEngine(**kwa)
    # init the searchlight with the queryengine
    return Searchlight(datameasure, queryengine=qe, roi_ids=center_ids,
                       **kwargs)
    
    
    

def to_file(fn,a):        
    with open(fn,'w') as f:
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def from_file(fn):
    print "Loading from %s" % fn
    with open(fn) as f:
        r=pickle.load(f)
    return r

def _sayhello():
    print "hello"
        
            
def _test_roi():
    vg=volgeom.VolGeom(None,None)
    ra=SparseVolumeAttributes(["voxel_ids","center_distances"],vg)
    ra.add(1,dict(voxel_ids=[1,2],center_distances=[.2,.3]))
    ra.add(2,dict(voxel_ids=[3,4,5],center_distances=[.2,.3,.5]))
    
    it=ra.get_attr_mapping("voxel_ids")
    for k,v in it.iteritems():
        print k, v, "=",it[k]
        
    print it.keys()
    
    fnout="/tmp/foo.pip"
    to_file(fnout,ra)
    
    print fnout
    
    rb=from_file(fnout)
    print rb
    
    for k,v in rb.get_attr_mapping("center_distances").iteritems():
        print k, v, "=",it[k]
    
if __name__ == "__main__":
    _test_roi()
    
    
                                     
        
    
    
        
     