import cPickle as pickle
import volgeom
from collections import Mapping

import nibabel as ni, numpy as np
import collections
import volgeom
import utils
import operator

from mvpa2.misc.neighborhood import IndexQueryEngine
from mvpa2.measures.searchlight import Searchlight

'''
This class is intended for general use of storing sparse attributes
associated with keys in a dictionary. 

Instantiation requires a set of labels (called sa_labels to mimick Dataset)
and every entry added has to be a dict with the keys matching sa_labels.  
'''
class SparseAttributes(object):
    def __init__(self, sa_labels):
        self._sa_labels = list(sa_labels)
        self.sa = dict()
        self.a = dict()

    def set(self, roi_label, roi_attrs_dict):
        if not roi_attrs_dict:
            roi_attrs_dict = None
        else:
            if not type(roi_attrs_dict) is dict:
                raise TypeError("roi attributes should be a dict, but is %r:\n%r" %
                                (type(roi_attrs_dict), roi_attrs_dict))

            # if sa_labels is set, check it has all the keys
            if set(self._sa_labels) != set(roi_attrs_dict):
                raise ValueError("Key set mismatch: %r != %r" %
                                 (self._sa_labels, roi_attrs_dict))

        self.sa[roi_label] = roi_attrs_dict

    def add(self, roi_label, roi_attrs_dict):
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
        roiattr = self.sa[roi_label]
        return roiattr[sa_label] if roiattr else None

    def get_tuple(self, roi_label, sa_labels=None):
        if sa_labels is None:
            sa_labels = tuple(self._sa_labels)

        roiattr = self.sa[roi_label]
        if not roiattr:
            return None

        vs = [roiattr[sa_label] for sa_label in sa_labels]

        return zip(*vs)


    def get_attr_mapping(self, roi_attr):
        '''Provides a dict-like object with lookup for a 
        single ROI attribute'''

        if not roi_attr in self.sa_labels():
            raise KeyError("attribute %r not in map" % roi_attr)

        return AttrMapping(self, roi_attr)

    def get_inv_attr_mapping(self, roi_attr):
        '''Provides an inverse mapping from get_attr_mapping
        This mapping is computed and returned as a new dict
        '''
        if isinstance(roi_attr, AttrMapping):
            forward = roi_attr
        else:
            forward = self.get_attr_mapping(roi_attr)

        backward = dict()
        for k, vs in forward.iteritems():
            for v in vs:
                if not v in backward:
                    backward[v] = []
                backward[v].append(k)

        return backward


    def __repr__(self):
        return ("SparseAttributes with %i entries, %i labels (%r)\nGeneral attributes: %r" %
                (len(self.sa), len(self._sa_labels), self._sa_labels, self.a.keys()))

class AttrMapping(Mapping):
    def __init__(self, cls, roi_attr):
        self._roi_attr = roi_attr
        self._cls = cls

    def __getitem__(self, key):
        return self._cls.get(key, self._roi_attr)

    def __len__(self):
        return len(self.__keys__())

    def __keys__(self):
        return list(self._cls.keys())

    def __iter__(self):
        return iter(self.__keys__())

def paired_common_attributes_count(sp_attrs, roi_attr):
    '''function to count how often the same attribute is shared across keys in
    the input sparse_attributes. Useful to count how many voxels are shared across 
    center nodes'''
    inv = sp_attrs.get_inv_attr_mapping(roi_attr)
    swp = lambda x, y: (x, y) if x < y else (y, x)

    d = dict()
    for k, vs in inv.iteritems():
        for i, x in enumerate(vs):
            for j, y in enumerate(vs):
                if i <= j:
                    continue
                s = swp(x, y)
                if not s in d:
                    d[s] = 0
                d[s] += 1
    return d

def voxel2nearest_node(sp_attrs, sort_by_label=[
                                    'center_distances',
                                    'grey_matter_position'],
                            sort_by_proc=[None, lambda x : abs(x - .5)],
                            return_label='lin_vox_idxs'):

    '''finds for each voxel the nearest node
    
    'nearest' in this sense depends on the criteria specified in the arguments.
    by default distances are first compared based on geodesic distance
    from lines connecting white/pial matter surfaces, and then relative
    position in the grey matter (.5 means just in between the two) 
    
    in a typical use case, the first input argument is the result
    from running voxel_selection
    
    added for swaroop
    
    TODO better documentation
    '''


    all_labels = sort_by_label + [return_label]
    sort_by_count = len(sort_by_label)

    id = lambda x:x # identity function
    if sort_by_proc is None:
        sort_by_proc = [None for _ in xrange(sort_by_count)]

    sort_by_proc = [id if f is None else f for f in sort_by_proc]


    sort_by_getter = lambda x: tuple([f(x[i]) for i, f in enumerate(sort_by_proc)])
    sort_by_getter_count = len(sort_by_proc)
    #key_getter = operator.itemgetter(*range(sort_by_count))
    value_getter = operator.itemgetter(sort_by_getter_count) # item after sort_by 

    node_idxs = sp_attrs.keys()
    vox2node_and_attrs = dict()

    for node_idx in node_idxs:
        attr_vals = sp_attrs.get_tuple(node_idx, all_labels)

        for attr_val in attr_vals:
            k, v = sort_by_getter(attr_val), value_getter(attr_val)

            if v in vox2node_and_attrs:
                n, attrs = vox2node_and_attrs[v]
                if k >= attrs:
                    continue

            vox2node_and_attrs[v] = (node_idx, k)

    n2v = dict((k, v[0]) for k, v in vox2node_and_attrs.iteritems())

    return n2v

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
    def __init__(self, sa_labels, volgeom):
        super(self.__class__, self).__init__(sa_labels)
        self.a['volgeom'] = volgeom

    """
    Returns a neighboorhood function
    
    Parameters
    ==========
    mask
        volume mask (usually from get_niftiimage_mask)
    voxel_ids_label
        label of voxel ids of neighboors
    
    
    Returns
    =======
    neighboorhood: SparseNeighborhood 
        neighboorhood function that can be used to access the 
        neighbors for searchlight centers stored in this instance
    """

    def get_neighborhood(self, mask=None, voxel_ids_label='lin_vox_idxs'):
        if mask:
            # take into account that masking means we have
            # to translate our center ids. We do so by creating a fresh
            # attribute mapping

            tp = type(mask)
            if not tp is np.ndarray:
                mask = mask.get_data()

            vg = self.get_volgeom()
            nv = vg.nv()
            mask = np.reshape(mask, (nv,))
            mask_idxs = np.nonzero(mask)[0]
            center_ids = self.keys() # these are only nodes with voxels associated

            if len(mask_idxs) < len(center_ids):
                raise ValueError("Too many center ids (%r > %r)" %
                                    (len(mask_idxs), len(center_ids)))

            attr = SparseVolumeAttributes([voxel_ids_label], vg)
            """ This is how I found this -- Andy Tues. Juy 17, 2012
            for i, center_id in enumerate(center_ids):
                vs = self.get(center_id, voxel_ids_label)
                attr.add(mask_idxs[i], {voxel_ids_label:vs})
            """
            ## (Andy Connoly July 2012) Trying the following change :
            for center_id in center_ids:
                vs = self.get(center_id, voxel_ids_label)
                attr.add(mask_idxs[center_id], {voxel_ids_label:vs})

#            for i, center_id in enumerate(center_ids):
#                vs = self.get(center_id, voxel_ids_label)
#                attr.add(mask_idxs[i], {voxel_ids_label:vs})

        else:
            attr = self

        return SparseNeighborhood(attr, voxel_ids_label)

    """
    Returns
    =======
    volgeom: volgem.VolGeom
        volume geometry stored in this instance
    """
    def get_volgeom(self):
        return self.a['volgeom']

    """
    Returns
    =======
    volgeom: volgem.VolGeom
        volume geometry stored in this instance
    """


    def get_linear_mask(self, voxel_ids_label='lin_vox_idxs', auto_grow=True):
        # if auto_grow=True, then we add enough positive values to the mask 
        # so that it reaches at least the number of self.keys().
        # This  is necessary when it is used in a searchlight
        vg = self.get_volgeom()

        linmask = np.zeros(shape=(vg.nv(),), dtype=np.int8)

        keys = self.keys()
        map = self.get_attr_mapping(voxel_ids_label)

        for key in keys:
            linmask[map[key]] = 1

        nmask = np.sum(linmask)
        nkeys = len(keys)
        delta = nkeys - nmask

        print "made maks, %d keys, %d in mask" % (len(keys), nmask)

        if auto_grow and delta > 0:
            # not enough values in the mask, have to add
            if nkeys > vg.nv():
                raise ValueError("%r keys but volume has too few voxels (%r)" %
                                 nkeys, vg.nv())
            print "Adding %r more" % delta
            for k in xrange(vg.nv()):
                if not linmask[k]:
                    linmask[k] = 2
                    delta -= 1
                if delta == 0:
                    break

        return linmask


    def get_niftiimage_mask(self, voxel_ids_label='lin_vox_idxs', auto_grow=True):
        vg = self.get_volgeom()
        shape = vg.shape()[:3]

        linmask = self.get_linear_mask(voxel_ids_label, auto_grow)
        mask = np.reshape(linmask, shape)

        img = ni.Nifti1Image(mask, vg.affine())
        return img



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

    def __init__(self, attr, voxel_ids_label='lin_vox_idxs'):
        if not voxel_ids_label in attr.sa_labels():
            raise ValueError("%r is not a valid key in %r" % (voxel_ids_label, attr))

        self._attr = attr
        self._voxel_ids_label = voxel_ids_label

    def __call__(self, coordinate):
        c_ijk = np.asanyarray(coordinate)[np.newaxis]

        if not c_ijk.shape == (1, 3):
            raise ValueError('Coordinate should be an length-3 iterable')

        vg = self._attr.a['volgeom']
        if not vg.ijkinvol(c_ijk): # illegal value (too big, probably), return nothing
            return tuple()

        c_lin = vg.ijk2lin(c_ijk)

        print c_lin, c_ijk

        if not c_lin in self._attr.keys(): # no nodes associated
            print "Not in keys!"
            return tuple()

        if len(c_lin) != 1:
            raise ValueError("No unique value for c_lin (found %d)"
                             "; something went wrong" % len(c_lin))

        a_lin = self._attr.get(c_lin[0], self._voxel_ids_label)
        if a_lin is None:
            return tuple()

        a_ijk = vg.lin2ijk(a_lin)

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




def to_file(fn, a):
    with open(fn, 'w') as f:
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)

def from_file(fn):
    with open(fn) as f:
        r = pickle.load(f)
    return r



