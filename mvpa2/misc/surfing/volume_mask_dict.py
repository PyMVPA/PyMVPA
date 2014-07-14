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

WiP.
TODO: make target to sources dictionary computations 'lazy'-only
compute the first time this is asked. XXX How to deal with __setstate__
and __getstate__ - have to flag somehow whether this mapping was present.
"""

__docformat__ = 'restructuredtext'

from collections import Mapping

import numpy as np

from mvpa2.base import externals
from mvpa2.misc.surfing import volgeom

from mvpa2.support.utils import deprecated

if externals.exists('nibabel'):
    import nibabel as nb
if externals.exists('h5py'):
    from mvpa2.base.hdf5 import h5save, h5load


class VolumeMaskDictionary(Mapping):
    """Collection of 3D volume masks, indexed by integers or strings.

    Voxels in a mask are represented sparsely, that is by linear indices in
    the [0, n-1] range, where n is the number of voxels in the volume.

    A typical use case is storing voxel selection results, which can
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
    """
    # TODO: docs for src2nbr and src2aux
    def __init__(self, vg, source, meta=None, src2nbr=None, src2aux=None):
        """Initialize a VolumeMaskDictionary

        Parameters
        ----------
        vg: volgeom.VolGeom or fmri_dataset-like or str
            data structure that contains volume geometry information.
        source: Surface.surf or numpy.ndarray or None
            structure that contains the geometric information of
            (the centers of) each mask. In the case of surface-searchlights this
            should be a surface used as the center for searchlights.
        meta: dict or None
            Optional meta data stored with this instance (such as searchlight
            radius and volumetric information). A use case is storing an instance
            and loading it later, and then checking whether the meta information
            is correct when it used to run a searchlight analysis.
        src2nbr: dict or None
            In a typical use case it contains a mapping from node center
            indices to lists of voxel indices.
        src2aux: dict or None
            In a typical use case it can contain auxiliary information such as
            distance of each voxel to each center.
        """
        self._volgeom = volgeom.from_any(vg)
        self._source = source


        self._src2nbr = dict() if src2nbr is None else src2nbr
        self._src2aux = dict() if src2nbr is None else src2aux

        self._meta = meta

        # this attribute is initially set to None
        # upon the first call that requires an inverse mapping
        # it is generated.
        self._lazy_nbr2src = None

    def __repr__(self, prefixes=[]):
        prefixes_ = ['vg=%r' % self._volgeom,
                    'source=%r' % self._source] + prefixes

        if not self._meta is None:
            prefixes_.append('meta=%r' % self._meta)

        if not self._src2nbr is None:
            prefixes_.append('src2nbr=%r' % self._src2nbr)
        if not self._src2aux is None:
            prefixes_.append('src2aux=%r' % self._src2aux)

        return "%s(%s)" % (self.__class__.__name__, ','.join(prefixes_))

    def __str__(self):
        return '%s(%d centers, volgeom=%s)' % (self.__class__.__name__,
                                               len(self._src2nbr),
                                               self._volgeom)

    def add(self, src, nbrs, aux=None):
        """Add a volume mask

        Parameters
        ----------
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
        """

        if not type(src) in [int, basestring]:
            # for now to avoid unhasbable type
            raise TypeError("src should be int or str")

        if src in self._src2nbr:
            raise ValueError('%s already in %s' % (src, self))


        self._src2nbr[src] = np.asarray(nbrs, dtype=np.int)

        if not self._lazy_nbr2src is None:
            self._add_target2source(src)

        if aux:
            n = len(nbrs)
            expected_keys = set(self.aux_keys())
            if expected_keys and (set(aux) != expected_keys):
                raise ValueError("aux label mismatch: %s != %s" %
                                (set(aux), expected_keys))
            for k, v in aux.iteritems():
                if not k in self._src2aux:
                    self._src2aux[k] = dict()

                # ensure that values have the same datatype for different keys
                if len(self._src2aux[k]) == 0:
                    v_dtype = None
                else:
                    v_dtype = next(self._src2aux[k].itervalues()).dtype

                if isinstance(v, (list, tuple, int, float, np.ndarray)):
                    v_arr = np.asanyarray(v, dtype=v_dtype).ravel()
                else:
                    raise ValueError('illegal type %s for %s' % (type(v), v))

                if len(v_arr) not in (n, 1):
                    raise ValueError('size mismatch: size %d != %d or 1' %
                                        (len(v_arr), n))

                self._src2aux[k][src] = v_arr



    def get_tuple_list(self, src, *labels):
        """Return a list of tuples with mask indices and/or aux information.

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
        """
        idxs = self[src]
        n = len(idxs)

        tuple_lists = []
        for label in labels:
            if label is None:
                tuple_list_elem = idxs.tolist()
            else:
                vs = self.get_aux(src, label)
                if len(vs) == 1:
                    tuple_list_elem = [vs[0]] * n
                else:
                    tuple_list_elem = vs.tolist()

            tuple_lists.append(tuple_list_elem)

        return zip(*tuple_lists)

    @deprecated("use .get_tuple_list instead")
    def get_tuple_list_dict(self, *labels):
        """Return a dictionary of mapping that maps each mask index
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
        """
        d = dict()
        for src in self.keys():
            d[src] = self.get_tuple_list(src, *labels)
        return d

    # XXX:  should not be logically 'get_indices'?
    #       while 'def get' could just return the mask?
    # YYY:  it's primary purpose is to store a mapping from
    #       node indices to lists of voxel indices. In that use case
    #       it would make sense to keep it as a mapping.
    #
    # XXX:  It overloads original Mapping.get which also
    #       had default... should it be the same here may be?
    # YYY:  I don't see any solution with a reasonable default.
    def get(self, src):
        """Return the linear voxel indices of a mask

        Parameters
        ----------
        src: int
            index of mask

        Returns
        -------
        idxs: list of int
            linear voxel indices indexed by src
        """
        return self._src2nbr[src].tolist()

    @deprecated("use .get_aux instead")
    def aux_get(self, src, label):
        return self.get_aux(src, label)

    def get_aux(self, src, label):
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
        return self._src2aux[label][src].tolist()

    # XXX:  get_aux_labels?
    # YYY:  aux is also a dictionary (actually a dictionary with dictionaries)
    #       so it seems more logical to use 'keys' instead of 'labels'.
    def aux_keys(self):
        '''Names of auxiliary labels

        Returns
        -------
        keys: list of str
            Names of auxiliary labels that are supported by get_aux
        '''
        return self._src2aux.keys()

    def _ensure_has_target2sources(self):
        '''Helper function to ensure that inverse mapping is set properly'''
        if not self._lazy_nbr2src:
            self._lazy_nbr2src = dict()
            for src in self.keys():
                self._add_target2source(src)

    def _add_target2source(self, src, targets=None):
        if targets is None:
            targets = self[src]

        contains = self.volgeom.contains_lin(np.asarray(targets))
        for i, target in enumerate(targets):
            if not contains[i]:
                raise ValueError("Target not in volume: %s" % target)
            if not target in self._lazy_nbr2src:
                self._lazy_nbr2src[target] = set()
            self._lazy_nbr2src[target].add(src)


    def target2sources(self, nbr):
        """Find the indices of masks that map to a linear voxel index

        Parameters
        ----------
        nbr: int
            Linear voxel index

        Returns
        -------
        srcs: list of int
            Indices i for which get(i) contains nbr
        """
        if type(nbr) in (list, tuple):
            return map(self.target2sources, nbr)

        self._ensure_has_target2sources()

        if not nbr in self._lazy_nbr2src:
            return None

        return self._lazy_nbr2src[nbr]

    def get_targets(self):
        """Return list of voxels that are in one or more masks

        Returns
        -------
        idxs: list of int
            Linear indices of voxels in one or more masks
        """
        self._ensure_has_target2sources()

        return sorted(self._lazy_nbr2src.keys())

    def _check_has_keys(self, keys=None, raise_=True):
        """Check that a list of keys is present; if not raise an error

        Parameters
        ----------
        keys: list or None
            List of keys that must be a subset of self.keys()
        raise_: boolean
            If True an error is raised if at least one key is not present
            in self.keys()

        Returns
        -------
        is_present: boolean
            If keys is None this function always returns True. Otherwise
            this function returns True if and only if every key in keys
            is present in self.keys(). If that is not the case and raise_
            is True and exception is raised.
        """
        if keys is None:
            return True

        missing_keys = set(keys).difference(set(self.keys()))
        n_missing = len(missing_keys)
        has_missing = n_missing > 0
        if has_missing and raise_:
                raise KeyError('%d keys (including "%s") not present' %
                                            (n_missing, missing_keys.pop()))
        return has_missing


    def get_mask(self, keys=None):
        """Return a mask for voxels that are included in one or more masks

        Parameters
        ----------
        keys: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all keys are used.

        Returns
        -------
        msk: np.ndarray
            Three-dimensional array with True for voxels that are
            included in one or more masks, and False elsewhere

        Notes
        -----
        When using surface-based searchlights, a use case of this function is
        to get the voxels that were associated with the searchlights in a
        subset of all nodes on a cortical surface.
        """
        self._check_has_keys(keys)
        self._ensure_has_target2sources()
        m_lin = np.zeros((self.volgeom.nvoxels, 1), dtype=np.int8)

        if keys is None:
            keys = self.keys()

        for key in keys:
            m_lin[self[key]] = 1

        return np.reshape(m_lin, self.volgeom.shape[:3])

    def get_nifti_image_mask(self, keys=None):
        """Return a NIfTI image with the voxels included in any mask

        Parameters
        ----------
        keys: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all keys are used.

        Returns
        -------
        msk: nibabel.Nifti1Image
            Nifti image where voxels have the value 1 for voxels that are
            included in one or more masks, and 0 elsewhere

        Notes
        -----
        When using surface-based searchlights, a use case of this function is
        to get the voxels that were associated with the searchlights in a
        subset of all nodes on a cortical surface.
        """
        # XXX:  my above dtype=np.int8 might kick back here, e.g.
        #       fslview iirc had difficulty with those
        # YYY:  should we change to a different data type, e.g. int32?
        return nb.Nifti1Image(self.get_mask(keys=keys), self.volgeom.affine)


    def get_voxel_indices(self, keys=None):
        """Returns voxel indices at least once selected

        Parameters
        ----------
        keys: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all keys are used.

        Returns
        -------
        voxel_indices: list of tuple
            List of triples with sub-voxel indices that were selected
            at least once since the initalization of this class.
            That is, a triple (i,j,k) referring to a voxel V is an element
            of voxel_indices iff there is at least one key k so that
            self.get(k) contains the linear index of voxel V.

        Notes
        -----
        When using surface-based searchlights, a use case of this function is
        to get the voxels that were associated with the searchlights in a
        subset of all nodes on a cortical surface.
        """
        self._check_has_keys(keys=keys)

        if keys is None:
            keys = self.keys()

        # get linear voxel indices
        lin_vox_set = set.union(*(set(self[k]) for k in keys))

        # convert to array
        lin_vox_arr = np.asarray(list(lin_vox_set))

        return map(tuple, self.volgeom.lin2ijk(lin_vox_arr))

    def get_dataset_feature_mask(self, ds, keys=None):
        """For a dataset return a mask of features that were selected
        at least once

        Parameters
        ----------
        ds: Dataset
            A dataset with field .fa.voxel_indices
        keys: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all keys are used.

        Returns
        -------
        mask: np.ndarray (boolean)
            binary mask with ds.nfeatures values, with True for features that
            were selected at least once once since the initalization of this class.
            That is, for a voxel (feature) with feature index i it holds that
            mask[I] is True iff there is at least one key k so that
            i in self.get(k).

        Notes
        -----
        When using surface-based searchlights, a use case of this function is
        to get the voxels that were associated with the searchlights in a
        subset of all nodes on a cortical surface.

        """
        # convert to tuples
        ds_voxel_indices = map(tuple, ds.fa.voxel_indices)
        sel_voxel_indices = map(tuple, self.get_voxel_indices(keys=keys))

        set_ds_voxel_indices = set(ds_voxel_indices)
        set_sel_voxel_indices = set(sel_voxel_indices)

        not_in_ds = set_sel_voxel_indices - set_ds_voxel_indices
        if not_in_ds:
            raise ValueError('Found %d voxel indices selected that were '
                             'not in dataset, first one is %s' %
                                (len(not_in_ds), not_in_ds.pop()))

        return np.asarray([d in sel_voxel_indices for d in ds_voxel_indices])

    def get_minimal_dataset(self, ds, keys=None):
        """For a dataset return only portion with features which were selected

        Parameters
        ----------
        ds: Dataset
            A dataset with field .fa.voxel_indices
        keys: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all keys are used.

        Returns
        -------
        Dataset
            A dataset containing features that were selected at least once

        Notes
        -----
        The rationale of this function is that voxel selection can be run
        first (without using a mask for a dataset), then the dataset
        can be reduced to contain only voxels that were selected by
        voxel selection
        """

        ds_mask = self.get_dataset_feature_mask(ds, keys=keys)
        return ds[:, ds_mask]

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return len(self.__keys__())

    def __keys__(self):
        return self._src2nbr.keys()

    def __iter__(self):
        return iter(self.__keys__())

    def __reduce__(self):
        return (self.__class__,
                (self._volgeom, self._source),
                self.__getstate__())

    @deprecated("should be used for testing compatibility only - "
                            "otherwise use .__reduce__ instead")
    def _reduce_legacy(self):
        return (self.__class__,
                (self._volgeom, self._source),
                self._getstate_legacy())


    def _getstate(self):
        s = (self._volgeom, self._source, self._meta, self._src2nbr, self._src2aux)
        return s

    @deprecated("should be used for testing compatibility only - "
                            "otherwise use ._getstate instead")
    def _getstate_legacy(self):
        return self._getstate()

    def __getstate__(self):
        # Note: due to issues with saving self_nbr2src, it is not returned
        # as part of the current state. instead it is derived when
        #  __setstate__ is called.

        # new as of Dec 2013: support more efficient storage method for
        # h5save/load

        s = self._getstate()
        # because of h5py issues when assigning values in the tuple s,
        # here a new tuple is created
        s3 = _dict_with_arrays2array_tuple(s[3])
        s4 = _dict_with_arrays2array_tuple(s[4])
        ss = s[:3] + (s3, s4)

        return ss


    def _setstate(self, s):
        # helper function that actually sets the state
        # it can be called by either __setstate__ or _setstate_legacy.
        # the rationale for a separate function is that it allows for
        # setting the state either with the up-to-date or the legacy method

        if len(s) == 4:
            # computatibilty thing: previous version (before Sep 12, 2013) did
            # not store meta
            self._volgeom, self._source, self._src2nbr, self._src2aux = s
            self._meta = False # signal that it is old (no meta information)
            warning('Using old (pre-12 Sep 2013) mapping - no meta data')
        else:
            self._volgeom, self._source, self._meta, self._src2nbr, self._src2aux = s


    @deprecated("should be used for testing compatibility only - "
                            "otherwise use ._setstate instead")
    def _setstate_legacy(self, s):
        # helper function to set the state as it was done prior to Dec 2013.
        # this function is defined separately so that unit tests can
        # override the __setstate__ method by this method and check for
        # compatibility
        self._setstate(s)


    def __setstate__(self, s):
        self._setstate(s)
        # new as of Dec 2013: use more efficient storage method for h5save/load
        self._src2nbr = _array_tuple2dict_with_arrays(self._src2nbr)
        self._src2aux = _array_tuple2dict_with_arrays(self._src2aux)


    def __eq__(self, other):
        """Compare this instance with another instance for equality

        Parameters
        ----------
        other: VolumeMaskDictionary
            instance with which the current instance is compared.

        Returns
        -------
        eq: bool
            True iff the current instance has the same layout as other,
            the same keys, and matching masks.
            This function does *not* consider auxiliary properties.
        """
        if not self.is_same_layout(other):
            return False

        if set(self.keys()) != set(other.keys()):
            return False

        for k in self.keys():
            if self[k] != other[k]:
                return False

        if set(self.aux_keys()) != set(other.aux_keys()):
            return False

        for lab in self.aux_keys():
            for k in self.keys():
                if self.get_aux(k, lab) != other.get_aux(k, lab):
                    return False

        if self.meta != False or other.meta != False:
            # both are 'new' ones with
            if self.meta != other.meta:
                return False

        return True

    @property
    def meta(self):
        '''Return meta information such as number of node indices

        Returns
        -------
        meta: dict or False
            dictionary with meta information, or False if no such information
            is present. The latter case is to be compatible with old-style
            instances loaded with h5load
        '''

        return dict(self._meta) if type(self._meta) is dict else self._meta

    def is_same_layout(self, other):
        '''Check whether another instance has the same spatial layout

        Parameters
        ----------
        other: VolumeMaskDictionary
            the instance that is compared to the current instance

        Returns
        -------
        same: boolean
            True iff the other instance has the same volume geometry
            and source as the current instance
        '''
        if not isinstance(other, self.__class__):
            return False

        return self.volgeom == other.volgeom and self.source == other.source

    # XXX:  shouldn't it be 'update'  mimicing dict.update?
    # YYY:  'update' does not raise an error if the key to be added is
    #       is already present; this method does.
    def merge(self, other):
        """Add masks from another instance

        Parameters
        ----------
        other: VolumeMaskDictionary
            The instance from which masks are added to the current one. The
            keys in the current and other instance should be disjoint, and
            auxiliary properties (if present) should have the same labels.
        """

        if not self.is_same_layout(other):
            raise ValueError("Cannot merge %s with %s" % (self, other))

        if not other:
            # nothing to add, so we're done
            return

        aks = self.aux_keys()
        if set(aks) != set(other.aux_keys()):
            if len(self.keys()) == 0:
                # current instance is empty, so use the keys from
                # the other (necessarily non-empty because of the check above)
                # instance
                aks = other.aux_keys()
            else:
                raise ValueError('Different keys in merge: %s != %s' %
                                (aks, other.aux_keys()))

# TODO: optimization in case either one or both already have the
#       inverse mapping from voxels to nodes
#       For now simply set everything to empty.
#        if self._lazy_nbr2src is None and not other._lazy_nbr2src is None:
#            self._ensure_has_target2sources()
#        elif other._lazy_nbr2src is None and not self._lazy_nbr2src is None:
#            other._ensure_has_target2sources()
#        elif not (other._lazy_nbr2src is None or self._lazy_nbr2src is None):
#            for k, vs in other._lazy_nbr2src.iteritems():
#                self._add_target2source(k, vs)

        self._lazy_nbr2src = None

        for k in other.keys():
            idxs = other[k]

            a_dict = dict()
            for ak in aks:
                a_dict[ak] = other.get_aux(k, ak)

            if not a_dict:
                a_dict = None

            self.add(k, idxs, a_dict)

    def xyz_target(self, ts=None):
        """Compute the x,y,z coordinates of one or more voxels

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
        """

        if ts is None:
            ts = list(self.get_targets())
        t_arr = np.reshape(np.asarray(ts), (-1,))
        return self.volgeom.lin2xyz(t_arr)

    def xyz_source(self, ss=None):
        """Computes the x,y,z coordinates of one or more mask centers

        Parameters
        ----------
        ss: list of int or None
            list of mask center indices for which coordinates should be
            computed. If ss is None, then coordinates for all masks that
            are mapped are computed.
            If is required that when the current instance was initialized,
            the source-argument was either a surf.Surface or a numpy.ndarray.
        """

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
        """Volume geometry information."""
        return self._volgeom

    @property
    def source(self):
        """Geometric information of (the centers of) each mask.

        In the case of surface-searchlights this
        should be a surface used as the center for searchlights.
        """
        return self._source

    def target2nearest_source(self, target, fallback_euclidean_distance=False):
        """Find the voxel nearest to a mask center

        Parameters
        ==========
        target: int
            linear index of a voxel
        fallback_euclidean_distance: bool (default: False)
            Whether to use a euclidean distance metric if target is not in
            any of the masks in this instance

        Returns
        =======
        src: int
            key index for the mask that contains target and is nearest to
            target. If target is not contained in any mask, then None is
            returned if fallback_euclidean_distance is False, and the
            index of the source nearest to target using a Euclidean distance
            metric is returned if fallback_euclidean_distance is True
        """
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
            if fallback_euclidean_distance:
                flat_srcs = self.keys()
            else:
                return None



        xyz_srcs = self.xyz_source(flat_srcs)
        d = volgeom.distance(xyz_srcs, xyz_trg)
        i = np.argmin(d)
        # d is a 2D array, get the row number with the lowest d
        source = flat_srcs[i // xyz_trg.shape[0]]

        return source

    def source2nearest_target(self, source):
        """Find the voxel nearest to a mask center

        Parameters
        ==========
        src: int
            mask index

        Returns
        =======
        target: int
            linear index of the voxel that is contained in the mask associated
            with src and nearest (Euclidean distance) to src.
        """

        trgs = self.__getitem__(source)
        trg_xyz = self.xyz_target(trgs)

        src_xyz = self.xyz_source(source)
        d = volgeom.distance(trg_xyz, src_xyz)
        i = np.argmin(d)

        return trgs[i / src_xyz.shape[0]]


def _dict_with_arrays2array_tuple(d):
    '''Helper: converts to a more efficient tuple-based representation

    This function converts a canonical representation of _src2nbr and
    _src2aux as dicts to a more efficient tuple-based representation.
    As a result, i/o operations using h5{load,save} are faster.

    Input: dict d where for each key k, each value v[k] is a numpy array
    Output: a tuple (keys, lengths, data) with each element a numpy array

    It holds that:
    - keys.tolist()==d.keys()
    - if k is the i-th element in d.keys(), then
          v[k]==data[offset+lengths[i]] where offset=np.sum(lenghts[:i])

    The 'inverse' of this function is _array_tuple2dict_with_arrays
    '''

    if d is None:
        return None
    if all(type(v) is dict for v in d.values()):
        # probably src2aux, so run recursively
        return dict((k, _dict_with_arrays2array_tuple(v))
                        for k, v in d.iteritems())

    keys = np.asarray(d.keys())

    lengths = np.asarray([len(d[key]) for key in keys])

    ntotal = np.sum(lengths)
    data = None # in case there are no keys
    pos = 0

    def _same_dtype(p, q):
        # helper function that returns whether p and q are of the same subtype.
        # this function returns True if, for example:
        #    p.dtype==np.float32 and q.dtype==np.float64
        pt, qt = pqt = p.dtype, q.dtype
        return pt == qt or np.issubdtype(pt, qt)


    for i, (key, length) in enumerate(zip(keys, lengths)):
        v = d[key]

        if i == 0:
            # allocate space for all data in d
            data = np.zeros((ntotal,), dtype=v.dtype)
        elif v.dtype != data.dtype:
            # ensure all values in the dict have the same datatype
            raise ValueError('Type mismatch for keys %s and %s: %s != %s' %
                                (keys[0], key, data.dtype, v.dtype))

        idxs = np.arange(length) + pos
        data[idxs] = v
        pos += length

    return keys, lengths, data


def _array_tuple2dict_with_arrays(kld):
        '''Helper: converts from a tuple-based representation to dictionaries

        This function converts a more efficient tuple-based representation
        of _src2nbr and _src2aux to canonical dicts

        Input: a tuple (keys, lengths, data) with each element a numpu array
        Output: dict d where for each key k, each value v[k] is a numpy array

        It holds that:
        - keys.tolist()==d.keys()
        - if k is the i-th element in d.keys(), then
              v[k]==data[offset+lengths[i]] where offset=np.sum(lenghts[:i])

        The 'inverse' of this function is _dict_with_arrays2array_tuple
        '''
        if kld is None:
            return None
        if type(kld) is dict:
            if all(type(v) in (tuple, dict) for v in kld.values()):
                # probably src2aux, so run recursively
                return dict((k, _array_tuple2dict_with_arrays(v))
                            for k, v in kld.iteritems())
            elif all(isinstance(v, np.ndarray) for v in kld.values()):
                # old-style mapping
                return kld
            else:
                raise ValueError('Unrecognized dict: %s' % kld)

        keys, lengths, data = kld

        # keys must be python int or str, not numpy int or str
        keys = keys.tolist()

        # space for output
        d = dict()

        pos = 0
        for key, length in zip(keys, lengths):
            d[key] = data[pos + np.arange(length)]
            pos += length

        if pos != data.size:
            raise ValueError('data size mismatch: expected %s, found %s' %
                                    (data.size, pos))
        return d


def from_any(s):
    """Load (if a string) or just return voxel selection

    Parameters
    ----------
    s: basestring or volume_mask_dict.VolumeMaskDictionary
        if a string it is assumed to be a file name and loaded using h5load. If
        a volume_mask_dict.VolumeMaskDictionary then it is returned.

    Returns
    -------
    r: volume_mask_dict.VolumeMaskDictionary
    """
    if isinstance(s, basestring):
        vs = h5load(s)
        return from_any(vs)
    elif isinstance(s, VolumeMaskDictionary):
        return s
    else:
        raise ValueError("Unknown type %s" % (type(s)))
