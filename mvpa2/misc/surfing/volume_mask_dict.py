# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dictionary (mapping) for storing several volume masks.
A typical use case is storing results from (surface-based) 
voxel selection.

@author: nick
"""


__docformat__ = 'restructuredtext'

import cPickle as pickle
from collections import Mapping

import nibabel as nb, numpy as np
import collections
import operator

from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs
from mvpa2.misc.surfing import volgeom

class VolumeMaskDictionary(Mapping):
    """A VolumeMaskDictionary is a collection of 3D volume masks, indexed
    by integers or strings. Voxels in a mask are represented sparsely, that
    is by linear indices in the range of 0 (inclusive) to n (exclusive), 
    where n is the number of voxels in the volume. 
    
    A typical use is case storing voxel selection results, which can
    subsequently be used for running searchlights. An alternative
    use case is storing a set of regions-of-interest masks.
    
    Because this class extends Mapping, it can be indexed as any other 
    mapping (like dicts). Currently masks cannot be removed, however, and
    adding masks is performed through the add(...) function rather than 
    item assignment.
    
    In some functions the terminology 'source' and 'target' is used. A source 
    refers to the label associated with masks (and can be used as a key 
    in the mapping), while a target is an element from the value associated
    in this mapping (i.e. typically a target is a linear voxel index).  
    
    Besides storing voxel indices, each mask can also have additional 
    'auxiliary' information associated, such as distance from the center
    or its relative position in grey matter.

    Parameters
    ----------
    vg: volgeom.VolGeom or fmri_dataset-like or str
        data structure that contains volume geometry information.
    source: Surface.surf or numpy.ndarray or None
        structure that contains the geometric information of 
        (the centers of) each mask. In the case of surface-searchlights this
        should be a surface used as the center for searchlights. 
    """
    def __init__(self, vg, source):
        self._volgeom = volgeom.from_any(vg)
        self._source = source

        self._src2nbr = dict()
        self._src2aux = dict()
        self._nbr2src = dict()
        self._aux_keys = set()

    def add(self, src, nbrs, aux=None):
        '''
        Adds a volume mask
        
        Parameters:
        src: int or str
            index or name of volume mask. src should not be already
            present in this dictionary
        nbrs: list of int
            linear voxel indices of the voxels in the mask
        aux: dict or None
            auxiliary properties associated with (the voxels in) the volume
            mask. If the current dictionary instance alraedy has stored
            auxiliary properties for other masks, then the set of keys in
            the current mask should be the same as for other masks. In 
            addition, the length of each value in aux should be either
            the number of elements in nbrs or one.
        '''

        if not type(src) in [int, basestring]:
            # for now to avoid unhasbable type
            raise TypeError("src should be int or str")

        if src in self._src2nbr:
            raise ValueError('%r already in %r' % (src, self))

        self._src2nbr[src] = nbrs

        n = len(nbrs)
        contains = self.volgeom.contains_lin(np.asarray(nbrs))
        for i, nbr in enumerate(nbrs):
            if not contains[i]:
                raise ValueError("Target not in volume: %s" % nbr)
            if not nbr in self._nbr2src:
                self._nbr2src[nbr] = set()
            self._nbr2src[nbr].add(src)

        if aux:
            if self._aux_keys and (set(aux) != self._aux_keys):
                raise ValuError("aux label mismatch: %r != %r" %
                                (set(aux), self._aux_keys))
            for k, v in aux.iteritems():
                if len(v) not in (n, 1):
                    raise ValueError('size mismatch: size %d != %d or 1' %
                                        (len(v), n))
                if not k in self._src2aux:
                    self._src2aux[k] = dict()
                self._src2aux[k][src] = v
                self._aux_keys.add(k)

    def get_tuple_list(self, src, *labels):
        '''
        Returns a list of tuples with mask indices and/or auxiliary 
        information.
        
        Parameters
        ----------
        src: int or str
            index of mask
        *labels: str or None
            List of labels to return. None refers to the voxel indices of the
            mask.
        
        Returns
        -------
        tuples: list of tuple
            N tuples each with len(labels) values, where N is the number of
            voxels in the mask indexed by src
        '''
        idxs = self[src]
        n = len(idxs)

        tuple_lists = []
        for label in labels:
            if label is None:
                tuple_lists.append(idxs)
            else:
                vs = self.aux_get(src, label)
                if len(vs) == 1:
                    vs = vs * n
                tuple_lists.append(vs)
        return zip(*tuple_lists)

    def get_tuple_list_dict(self, *labels):
        '''Returns a dictionary mapping that maps each mask index
        to tuples with mask indices and/or auxiliary 
        information.
        
        Parameters
        ----------
        *labels: str or None
            List of labels to return. None refers to the voxel indices of the
            mask.
            
        tuple_dict: dict
            a mapping so that 
            get_tuple_list(s, labels)==get_tuple_list_dict(labels)[s]
        '''
        d = dict()
        for src in self.keys():
            d[src] = self.get_tuple(src, *labels)
        return d

    def get(self, src):
        '''Returns the linear voxel indices of a mask
        
        Parameters
        ----------
        src: int
            index of mask
        
        Returns
        -------
        idxs: list of int
            linear voxel indices indexed by src
        '''
        return list(self._src2nbr[src])

    def aux_get(self, src, label):
        '''Auxiliary information of a mask
        
        Parameters
        ----------
        src: int
            index of mask
        label: str
            label of auxiliary information
            
        Returns
        -------
        vals: list
            auxiliary information labelled label for mask src
        '''
        labels = self.aux_keys()
        if not label in labels:
            raise ValueError("%s not in %r" % (label, labels))
        if not label in self._src2aux:
            msg = ("Mismatch for key %r" if src in self._src2nbr
                                        else "Unknown key %r")
            raise ValueError((msg + ', label %r') % (src, label))
        return self._src2aux[label][src]

    def aux_keys(self):
        '''Names of auxiliary labels
        
        Returns
        -------
        keys: list of str
            Names of auxiliary labels that are supported by aux_get
        '''
        return list(self._aux_keys)

    def target2sources(self, nbr):
        '''Finds the indices of masks that map to a linear voxel index
        
        Parameters
        ----------
        nbr: int
            Linear voxel index
        
        Returns
        -------
        srcs: list of int
            Indices i for which get(i) contains nbr
        '''
        if type(nbr) in (list, tuple):
            return map(self.target2sources, nbr)

        if not nbr in self._nbr2src:
            return None

        return self._nbr2src[nbr]

    def get_targets(self):
        '''The list of voxels that are in one or more masks
        
        Returns
        idxs: list of int
            Linear indices of voxels in one or more masks
        '''
        return sorted(self._nbr2src.keys())

    def get_mask(self):
        '''A mask of voxels that are included in one or more masks
        
        Returns
        -------
        msk: np.ndarray
            Three-dimensional array with the value 1 for voxels that are
            included in one or more masks, and 0 elsewhere
        '''
        m_lin = np.zeros((self.volgeom.nvoxels, 1))
        for src, nbrs in self._src2nbr.iteritems():
            for nbr in nbrs:
                m_lin[nbr] = 1

        return np.reshape(m_lin, self.volgeom.shape[:3])

    def get_nifti_image_mask(self):
        '''
         A nifti image indicating voxels that are included in one or more 
         masks
        
        Returns
        -------
        msk: nibabel.Nifti1Image
            Nifti image where voxels have the value 1 for voxels that are
            included in one or more masks, and 0 elsewhere
        '''
        return nb.Nifti1Image(self.get_mask(), self.volgeom.affine)

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.__keys__)

    def __keys__(self):
        return list(self._src2nbr)

    def __iter__(self):
        return iter(self.__keys__())

    def __reduce__(self):
        return (self.__class__,
                (self._volgeom, self._source),
                self.__getstate__())

    def __getstate__(self):
        return (self._volgeom, self._source, self._src2nbr,
                self._src2aux, self._nbr2src)

    def __setstate__(self, s):
        self._volgeom, self._source, self._src2nbr, \
                                self._src2aux, self._nbr2src = s

    def __eq__(self, other):
        if not self.same_layout(other):
            return False

        if set(self.keys()) != set(other.keys()):
            return False

        for k in self.keys():
            if self[k] != other[k]:
                return False

        return True

    def same_layout(self, other):
        '''
        Computers whether another instance has the same spatial layout
        
        Parameters
        ----------
        other: VolumeMaskDictionary
            the instance that is compared to the current instance
        
        Returns
        -------
        same: boolean
            True iif the other instance has the same volume geometry
            and source as the current instance
        '''
        if not isinstance(other, self.__class__):
            return False

        return self.volgeom == other.volgeom and self.source == other.source

    def merge(self, other):
        '''
        Adds masks from another instance 
        
        Parameters
        ----------
        other: VolumeMaskDictionary
            The instance from which masks are added to the current one. The
            keys in the current and other instance should be disjoint, and
            auxiliary properties (if present) should have the same labels.
        '''

        if not self.same_layout(other):
            raise ValueError("Cannot merge %r with %r" % (self, other))

        aks = self.aux_keys()

        for k in other.keys():
            idxs = other[k]

            a_dict = dict()
            for ak in aks:
                a_dict[ak] = other.aux_get(k, ak)

            if not a_dict:
                a_dict = None

            self.add(k, idxs, a_dict)

    def xyz_target(self, ts=None):
        '''Computes the x,y,z coordinates of one or more voxels
        
        Parameters
        ----------
        ts: list of int or None
            list of voxels for which coordinates should be computed. If 
            ts is None, then coordinates for all voxels that are mapped
            are computed
        
        Returns
        -------
        xyz: numpy.ndarray
            Array with size len(ts) x 3 with x,y,z coordinates
        '''

        if ts is None:
            ts = list(self.get_targets())
        t_arr = np.reshape(np.asarray(ts), (-1,))
        return self.volgeom.lin2xyz(t_arr)

    def xyz_source(self, ss=None):
        '''Computes the x,y,z coordinates of one or more mask centers
        
        Parameters
        ----------
        ss: list of int or None
            list of mask center indices for which coordinates should be 
            computed. If ss is None, then coordinates for all masks that 
            are mapped are computed.
            If is required that when the current instance was initialized,
            the source-argument was either a surf.Surface or a numpy.ndarray.
        '''

        # TODO add dataset and volgeom support
        coordinate_labels = [None, 'vertices', 'coordinates']
        coordinates = None
        for coordinate_label in coordinate_labels:
            s = self.source
            if coordinate_label and hasattr(s, coordinate_label):
                s = getattr(s, coordinate_label)
            if isinstance(s, np.ndarray):
                coordinates = s

        if coordinates is None:
            raise ValueError("Cannot find coordinates in %r" % self.source)

        if ss is None:
            ss = self.keys()
        if not isinstance(ss, np.ndarray):
            if type(ss) is int:
                ss = [ss]
            ss = np.asarray(list(ss)).ravel()

        return coordinates[ss]

    @property
    def volgeom(self):
        return self._volgeom

    @property
    def source(self):
        return self._source

    def target2nearest_source(self, target, fallback_euclidian_distance=False):
        targets = []
        if type(target) in (list, tuple):
            for t in target:
                targets.append(t)
        else:
            targets = [target]

        xyz_trg = self.xyz_target(np.asarray(targets))

        src = self.target2sources(targets)
        flat_srcs = []
        for s in src:
            if s:
                for j in s:
                    flat_srcs.append(j)

        if not flat_srcs:
            if fallback_euclidian_distance:
                flat_srcs = self.keys()
            else:
                return None

        xyz_srcs = self.xyz_source(flat_srcs)
        d = volgeom.distance(xyz_srcs, xyz_trg)
        i = np.argmin(d)

        # d is a 2D array, get the row number with the lowest d
        source = flat_srcs[i / xyz_trg.shape[0]]

        return source

    def source2nearest_target(self, source):
        trgs = self.__getitem__(source)
        trg_xyz = self.xyz_target(trgs)

        src_xyz = self.xyz_source(source)
        d = volgeom.distance(trg_xyz, src_xyz)
        i = np.argmin(d)

        return trgs[i / src_xyz.shape[0]]


def save(fn, attr):
    with open(fn, 'w') as f:
        pickle.dump(attr, fn, protocol=pickle.HIGHEST_PROTOCOL)

def read(fn):
    with open(fn) as f:
        r = pickle.load(fn)
    return r
