import cPickle as pickle
import volgeom
from collections import Mapping

class SparseROIAttributes(object):
    def __init__(self,sa_labels):
        self._sa_labels=list(sa_labels)
        self.sa=dict()
        self.a=dict()
        
    def add_roi_dict(self,roi_label,roi_attrs):
        # enesoure roi_attrs is a dict
        if not type(roi_attrs) is dict:
            raise TypeError("roi attributes should be a dict")
        
        # if sa_labels is set, check it has all the keys
        if set(self._sa_labels)!=set(roi_attrs):
            raise ValueError("Key set mismatch: %r != %r" % 
                             (self._sa_labels,roi_attrs))
    
        self.sa[roi_label]=roi_attrs
    
    def sa_labels(self):
        return list(self._sa_labels)
    
    def keys(self):
        return self.sa.keys()
    
    def get(self, roi_label, sa_label):
        return self.sa[roilabel][sa_label]
    
    def get_attr_mapping(self, roi_attr):
        class AttrMapping(Mapping):
            def __init__(self,cls,roi_attr):
                self._roi_attr=roi_attr
                self._cls=cls
                
            def __getitem__(self,key):
                return self._cls.sa[key][self._roi_attr]
            
            def __len__(self):
                return len(self.__keys__())
            
            def __keys__(self):
                return list(self._cls.keys())
            
            def __iter__(self):
                return iter(self.__keys__())
            
        return AttrMapping(self,roi_attr) 
    
    def __repr__(self):
        return ("SparseAttributes with %i entries, %i labels (%r)" % 
                (len(self.sa), len(self._sa_labels), self._sa_labels))
        

class SparseVolumeMasks(SparseROIAttributes):
    def __init__(self,sa_labels,volgeom):
        super(self.__class__,self).__init__(sa_labels)
        self.a['volgeom']=volgeom
        
    def get_searchlight(self):
        return SparseSearchlight(self)
        

class SparseSearchlight():
    def __init__(self,attr,voxel_ids_label='voxel_idxs'):
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
        
        if not c_lin in self._attr.sa.keys(): # no nodes associated
            return tuple()
        
        assert len(c_lin)==1
        
        a_lin=self._attr.get[c_lin[0]][self._voxel_ids_label]
        a_ijk=vg.lin2ijk(a_lin)
        
        a_tuples = [tuple(p) for p in a_ijk]
        
        return a_tuples
    

def to_file(fn,a):        
    with open(fn,'w') as f:
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def from_file(fn):
    with open(fn) as f:
        r=pickle.load(f)
    return r
        
            
def _test_roi():
    vg=volgeom.VolGeom(None,None)
    ra=SparseVolumeMasks(["voxel_ids","center_distances"],vg)
    ra.add_roi_dict(1,dict(voxel_ids=[1,2],center_distances=[.2,.3]))
    ra.add_roi_dict(2,dict(voxel_ids=[3,4,5],center_distances=[.2,.3,.5]))
    
    it=ra.get_attr_mapping("voxel_ids")
    for k,v in it.iteritems():
        print k, v, "=",it[k]
        
    print it.keys()
    
    fnout="/tmp/foo.pip"
    to_file(fnout,ra)
    
    rb=from_file(fnout)
    print rb
    
    for k,v in rb.get_attr_mapping("center_distances").iteritems():
        print k, v, "=",it[k]
    
if __name__ == "__main__":
    _test_roi()
    
    
                                     
        
    
    
        
     