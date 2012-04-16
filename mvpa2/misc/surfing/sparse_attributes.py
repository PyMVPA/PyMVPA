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

'''
This class is intended for general use of storing sparse attributes
associated with keys in a dictionary. 

Instantiation requires a set of labels (called sa_labels to mimick Dataset)
and every entry added has to support  
'''
class SparseAttributes(object):
    def __init__(self,sa_labels):
        self._sa_labels=list(sa_labels)
        self.sa=dict()
        self.a=dict()
        
    def set(self,roi_label,roi_attrs_dict):
        if not roi_attrs_dict:
            roi_attrs_dict=None
        else:
            if not type(roi_attrs_dict) is dict:
                raise TypeError("roi attributes should be a dict, but is %r:\n%r" %
                                (type(roi_attrs_dict),roi_attrs_dict))
            
            # if sa_labels is set, check it has all the keys
            if set(self._sa_labels)!=set(roi_attrs_dict):
                raise ValueError("Key set mismatch: %r != %r" % 
                                 (self._sa_labels,roi_attrs_dict))
        
        self.sa[roi_label]=roi_attrs_dict
        
    def add(self,roi_label,roi_attrs_dict):
        if roi_label in self.sa.keys():
            raise ValueError("name clash: key %s already present" % roi_label)
        self.set(roi_label, roi_attrs_dict)
    
    def sa_labels(self):
        return list(self._sa_labels)
    
    def keys(self):
        return filter(lambda x : not self.sa[x] is None, self.all_keys())
    
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
    """
    Sparse attributes with volume geometry. 
    This class can store the result of surface-based voxel selection.
    
    Parameters
    ==========
    sa_labels: list
        labels of attributes stored in this instance
    volgeom: volgeom.VolGeom
        volume geometry
    """ 
    def __init__(self,sa_labels,volgeom):
        super(self.__class__,self).__init__(sa_labels)
        self.a['volgeom']=volgeom
    
    """
    Returns a neighboorhood function
    
    Parameters
    ==========
    voxel_ids_label
        label of voxel ids of neighboors
    
    Returns
    =======
    neighboorhood: SparseNeighborhood 
        neighboorhood function that can be used to access the 
        neighbors for searchlight centers stored in this instance
    """
    
    def get_neighborhood(self,voxel_ids_label='lin_vox_idxs'):
        return SparseNeighborhood(self,voxel_ids_label)
    
    """
    Returns
    =======
    volgeom: volgem.VolGeom
        volume geometry stored in this instance
    """
    def get_volgeom(self):
        return self.a['volgeom']
        

class SparseNeighborhood():
    """
    Defines a neighborhood; the typical use case is surface nodes on the 
    surface, each of which has a set of surrounding voxels associated with it.
    
    Parameters
    ----------
    attr: SparseAttributes
        Holds the neighborhood information
    voxel_ids_label: str
        The label by which neighboors are stored. attr.get[i,voxel_ids_label]
        should return a list with voxels around center node i.
    
    Returns
    -------
    neighboorhood: callable
        This function can be passed as the neighborhood in the function
        sparse_attributes.searchlight()
    """ 
       
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
    with open(fn) as f:
        r=pickle.load(f)
    return r

def _sayhello():
    print "hello"
        
            
def _test_roi():
    sa=SparseAttributes(["lin_vox_idxs","center_distances"])
    sa.add(1,dict(lin_vox_idxs=[1,2],center_distances=[.2,.3]))
    sa.add(2,dict(lin_vox_idxs=[3,4,5],center_distances=[.2,.3,.5]))
    sa.add(6,dict(lin_vox_idxs=[3,4,8,9],center_distances=[.1,.3,.2,.2]))
    sa.a['some general attribute']='voxel selection example'
    
    print sa
    
    it=sa.get_attr_mapping("lin_vox_idxs")
    for k,v in it.iteritems():
        print "%r -> %r==%r" % (k, v, it[k])
        
    fn="/tmp/foo.pickle"
    to_file(fn,sa)
    sb=from_file(fn)
    
    print sb
    
    it2=sb.get_attr_mapping("center_distances")
    for k,v in it2.iteritems():
        print "%r -> distances=%r, idxs==%r)" % (k, v, it[k])
    
if __name__ == "__main__":
    _test_roi()
    
    
                                     
        
    
    
        
     