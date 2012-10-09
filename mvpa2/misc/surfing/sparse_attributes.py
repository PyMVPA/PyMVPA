# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
This class is intended for general use of storing sparse attributes
associated with keys in a dictionary.

Instantiation requires a set of labels (called sa_labels to mimick Dataset)
and every entry added has to be a dict with the keys matching sa_labels.
It is assumed, but not enforced, that the different values in such a dict 
have the same number of elements.

The rationale is to make a general class that associates keys of ROIs with
properties of voxels (their index, distance from center, position in grey 
matter, etc). ROIs can in principle be either from a volume or a
surface searchlight, or manually drawn in e.g. AFNI's SUMA.

TODO: consider whether we use a standard sa_label, such as the one
used for linear voxel indices.
TODO: consider using constants frmo surf_voxel_selection   
"""

import cPickle as pickle
from collections import Mapping

import nibabel as nb, numpy as np
import collections
import operator

from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs

from mvpa2.misc.neighborhood import IndexQueryEngine
from mvpa2.measures.searchlight import Searchlight

class SparseAttributes(object):
    def __init__(self, sa_labels, sa=None, a=None):
        self._sa_labels = list(sa_labels)
        self.sa = dict() if sa is None else sa
        self.a = dict() if a is None else a

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
        if roi_label in self.keys:
            raise ValueError("name clash: key %s already present" % roi_label)
        self.set(roi_label, roi_attrs_dict)

    @property
    def sa_labels(self):
        return list(self._sa_labels)

    @property
    def keys(self):
        return filter(lambda x : not self.sa[x] is None, self.all_keys)

    @property
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

        if not roi_attr in self.sa_labels:
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

    def __repr__(self, prefixes=[]):
        # do no bother for complete repr of the beast for now
        # prefixes = prefixes + _repr_attrs(self, ['sa', 'a'])
        suffix = ''
        if len(prefixes):
            suffix = ', '.join([''] + prefixes)
        return "%s(%r%s) #" % (self.__class__.__name__,
                           self._sa_labels, suffix)

    def __str__(self):
        return ("SparseAttributes with %i entries, %i labels (%r)\nGeneral attributes: %r" %
                (len(self.sa), len(self._sa_labels), self._sa_labels, self.a.keys()))

    def __eq__(self, other):
        if not isinstance(other, SparseAttributes):
            return False

        if set(self.keys) != set(other.keys):
            return False
        if set(self.all_keys) != set(other.all_keys):
            return False
        if set(self.sa_labels) != set(other.sa_labels):
            return False

        labs = self.sa_labels
        for k in self.keys:
            if self.get_tuple(k, labs) != other.get_tuple(k, labs):
                return False

        return True

class SparseVolumeAttributes(SparseAttributes):
    """
    Sparse attributes with volume geometry. 
    This class can be used to store the result of surface-based voxel 
    selection.
    
    TODO: consider whether we integrate this with SparseAttributes
    
    Parameters
    ==========
    sa_labels: list
        labels of attributes stored in this instance
    volgeom: volgeom.VolGeom
        volume geometry
    """
    def __init__(self, sa_labels, volgeom, **kwargs):
        super(self.__class__, self).__init__(sa_labels, **kwargs)
        self.a['volgeom'] = volgeom

    def __repr__(self, prefixes=[]):
        return super(SparseVolumeAttributes, self).__repr__(
            prefixes + ['volgeom=%r' % self.a['volgeom']])

    @property
    def volgeom(self):
        """
        Returns
        =======
        volgeom: volgem.VolGeom
            volume geometry stored in this instance
        """

        return self.a['volgeom']



    def get_linear_mask(self, sa_label='linear_voxel_indices'):
        vg = self.volgeom
        linear_mask = np.zeros((vg.nvoxels,), dtype=np.int32)

        for k in self.keys:
            vox_idxs = self.get(k, sa_label)
            linear_mask[vox_idxs] += 1

        return linear_mask


    def get_mask(self, sa_label='linear_voxel_indices'):
        linear_mask = self.get_linear_mask(sa_label)
        vg = self.volgeom
        volsize = vg.shape[:3]
        return np.reshape(linear_mask, volsize)

    def get_masked_instance(self, mask, sa_label='linear_voxel_indices',
                            mask_function=lambda x:x != 0):
        '''Allows for post-hoc application of a mask on voxels
        This is not very efficient as we copy all the data over
        TODO: consider allowing this in initial voxel selection
        
        NOTE: this function is not suitable if voxel selection was run with
        a fixed number of voxels, as voxels are most likely to masked out
        with this function
        '''
        if not type(mask) is np.ndarray:
            raise TypeError('Only numpy arrays are supported (for now)')

        vg = self.volgeom
        if mask.size != vg.nvoxels:
            raise ValueError('Mask size (%d) mismatches volgeom size (%d)' %
                              (mask.size, vg.nvoxels))

        # make a linear mask - so that indexing works
        mask_lin = np.reshape(mask, (mask.size,))
        labs = self.sa_labels
        if not sa_label in labs:
            raise ValueError("Illegal sa_label: %s" % sa_label)
        # make a new instance
        attr = SparseVolumeAttributes(labs, self.volgeom)

        # copy values over - except if no voxels survive the mask
        for k in self.keys:
            idxs = self.get(k, sa_label)
            mask_idxs = mask_function(mask_lin[idxs])
            if np.sum(mask_idxs) == 0:
                continue # no indices in mask - continue

            d = dict()
            for lab in labs:
                d[lab] = self.get(k, lab)[mask_idxs]
            attr.add(k, d)
        return attr

    def map_all_to_masked(self, sa_label='linear_voxel_indices'):
        '''Creates a mapping from all voxels to those in the mask
        
        Typical use case is applying this to the output of voxel2nearest_node.
        
        Returns
        -------
        all2masked: np.ndarray (P-vector)
            all2masked[i]==j means that voxel with in linear index i is the
            j-th voxel that survives the mask, if j>0. If j<0, then there
        '''
        msk = self.get_linear_mask(sa_label)
        msk_nonzero = np.nonzero(msk)

        a2m = dict()
        for i, v in msk_nonzero:
            a2m[v] = i


        n = m.shape[0]
        n_nonzero = np.sum(msk_nonzero)

        a2m = np.zeros((n,), dtype=np.int) - 1
        a2m[msk_nonzero] = np.arange(n_nonzero)

        return a2m





class AttrMapping(Mapping):
    def __init__(self, cls, roi_attr):
        self._roi_attr = roi_attr
        self._cls = cls

    def __getitem__(self, key):
        return self._cls.get(key, self._roi_attr)

    def __len__(self):
        return len(self.__keys__)

    def __keys__(self):
        return list(self._cls.keys)

    def __iter__(self):
        return iter(self.__keys__())

def paired_common_attributes_count(sp_attrs, roi_attr):
    '''function to count how often the same attribute is shared across keys in
    the input sparse_attributes. Useful to count how many voxels are shared across 
    pairs of center nodes'''
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
                            return_label='linear_voxel_indices',
                            apply_mask=False):

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

    sort_by_proc = [f or id for f in sort_by_proc]


    sort_by_getter = lambda x: tuple([f(x[i]) for i, f in enumerate(sort_by_proc)])
    sort_by_getter_count = len(sort_by_proc)
    #key_getter = operator.itemgetter(*range(sort_by_count))
    value_getter = operator.itemgetter(sort_by_getter_count) # item after sort_by 

    node_idxs = sp_attrs.keys
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

    if apply_mask:
        if isinstance(apply_mask, basestring):
            msk = sa_attr.get_linear_mask(apply_mask)
        else:
            msk = sa_attr.get_linear_mask()

        n = msk.shape[0]
        nz = np.nonzero(msk)
        nm = len(nm)

        all2inmask = np.zeros((n,), dtype=np.int)
        all2inmask[nz] = np.arange(nm)

        ks = n2v.keys()
        for k in ks:
            n2v[k] = all2inmask[n2v[k]]

    return n2v

def _get_redundancy_statistics(attr, sa_label='linear_voxel_indices'):
    m = attr.get_attr_mapping(sa_label)

    vs = map(set, m.values())
    n = len(vs)

    r = 0 # number of redundant items
    for i in xrange(n):
        # see if item i is redundant (i.e. covered by another item)
        for j in xrange(i + 1, n):
            if vs[i] == vs[j]:
                r += 1
                break


    return ('%d items, %d unique (redundancy %.1f%%) for "%s"' %
                                (n, r, 100. * r / n, sa_label))


def to_file(fn, a):
    '''
    Stores attributes in a file
    
    Parameters
    ----------
    fn: str
        Output filename
    a: SparseAttributes
        attributes to be stored
    '''

    with open(fn, 'w') as f:
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)

def from_file(fn):
    '''
    Reads attributes from a file
    
    Parameters
    ----------
    fn: str
        Input filename
    
    Returns
    -------
    a: SparseAttributes
        attributes to be stored
    '''

    with open(fn) as f:
        r = pickle.load(f)
    return r



