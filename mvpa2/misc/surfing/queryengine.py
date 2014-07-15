# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""QueryEngine for querying feature ids based on the surface nodes

References
----------
NN Oosterhof, T Wiestler, PE Downing, J Diedrichsen (2011). A comparison of volume-based
and surface-based multi-voxel pattern analysis. Neuroimage, 56(2), pp. 593-600

'Surfing' toolbox: http://surfing.sourceforge.net
(and the associated documentation)
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.dochelpers import _repr_attrs

from mvpa2.base.dataset import AttrDataset
from mvpa2.misc.neighborhood import QueryEngineInterface

from mvpa2.misc.surfing import volgeom, surf_voxel_selection
from mvpa2.base import warning


class SurfaceQueryEngine(QueryEngineInterface):
    '''
    Query-engine that maps center nodes to indices of features
    (nodes) that are near each center node.

    This class is for mappings from surface to surface features;
    for mappings from surface to voxel features, use
    SurfaceVerticesQueryEngine.
    '''

    def __init__(self, surface, radius, distance_metric='dijkstra',
                    fa_node_key='node_indices'):
        '''Make a new SurfaceQueryEngine

        Parameters
        ----------
        surface: surf.Surface or str
            surface object, or filename of a surface
        radius: float
            size of neighborhood.
        distance_metric: str
            'euclidean' or 'dijkstra' (default).
        fa_node_key: str
            Key for feature attribute that contains node indices
            (default: 'node_indices').

        Notes
        -----
        After training this instance on a dataset and calling it with
        self.query_byid(vertex_id) as argument,
        '''
        self.surface = surface
        self.radius = radius
        self.distance_metric = distance_metric
        self.fa_node_key = fa_node_key
        self._vertex2feature_map = None

        allowed_metrics = ('dijkstra', 'euclidean')
        if not self.distance_metric in allowed_metrics:
            raise ValueError('distance_metric %s has to be in %s' %
                                    (self.distance_metric, allowed_metrics))

        if self.distance_metric == 'dijkstra':
            # Pre-compute neighbor information (and ignore the output).
            surface.neighbors

    def __repr__(self, prefixes=[]):
        return super(SurfaceQueryEngine, self).__repr__(
                   prefixes=prefixes
                   + _repr_attrs(self, ['surface'])
                   + _repr_attrs(self, ['radius'])
                   + _repr_attrs(self, ['distance_metric'],
                                   default='dijkstra')
                   + _repr_attrs(self, ['fa_node_key'],
                                   default='node_indices'))

    def __reduce__(self):
        return (self.__class__, (self.surface,
                                 self.radius,
                                 self.distance_metric,
                                 self.fa_node_key),
                            dict(_vertex2feature_map=self._vertex2feature_map))

    def __str__(self):
        return '%s(%s, radius=%s, distance_metric=%s, fa_node_key=%s)' % \
                                               (self.__class__.__name__,
                                                self.surface,
                                                self.radius,
                                                self.distance_metric,
                                                self.fa_node_key)

    def _check_trained(self):
        if self._vertex2feature_map is None:
            raise ValueError('Not trained on dataset: %s' % self)


    @property
    def ids(self):
        self._check_trained()
        return self._vertex2feature_map.keys()

    def untrain(self):
        self._vertex2feature_map = None

    def train(self, ds):
        '''
        Train the queryengine

        Parameters
        ----------
        ds: Dataset
            dataset with surface data. It should have a field
            .fa.node_indices that indicates the node index of each
            feature.
        '''

        fa_key = self.fa_node_key
        nvertices = self.surface.nvertices
        nfeatures = ds.nfeatures

        if not fa_key in ds.fa.keys():
            raise ValueError('Attribute .fa.%s not found.', fa_key)

        vertex_ids = ds.fa[fa_key].value.ravel()

        # check that vertex_ids are not outside 0..nfeatures
        delta = np.setdiff1d(vertex_ids, np.arange(nvertices))

        if len(delta):
            raise ValueError("Vertex id '%s' found that is not in "
                             "np.arange(%d)" % (delta[0], nvertices))

        # vertex_ids can have multiple occurences of the same node index
        # for different features, hence use a list.
        # initialize each vertex with an empty list
        self._vertex2feature_map = v2f = dict((vertex_id, list())
                                            for vertex_id in xrange(nvertices))

        for feature_id, vertex_id in enumerate(vertex_ids):
            v2f[vertex_id].append(feature_id)


    def query(self, **kwargs):
        raise NotImplementedError


    def query_byid(self, vertex_id):
        '''
        Return feature ids of features near a vertex

        Parameters
        ----------
        vertex_id: int
            Index of vertex (i.e. node) on the surface

        Returns
        -------
        feature_ids: list of int
            Indices of features in the neighborhood of the vertex indexed
            by 'vertex_id'
        '''
        self._check_trained()

        if vertex_id < 0 or vertex_id >= self.surface.nvertices or \
                        round(vertex_id) != vertex_id:
            raise KeyError('vertex_id should be integer in range(%d)' %
                                                self.surface.nvertices)

        nearby_nodes = self.surface.circlearound_n2d(vertex_id,
                                                    self.radius,
                                                    self.distance_metric)

        v2f = self._vertex2feature_map
        return sum((v2f[node] for node in nearby_nodes), [])



class SurfaceVerticesQueryEngine(QueryEngineInterface):
    '''
    Query-engine that maps center nodes to indices of features
    (voxels) that are near each center node.

    In a typical use case such an instance is generated using
    the function 'disc_surface_queryengine'

    This class is for mappings from surface to voxel features;
    for mappings from surface to surface features, use
    SurfaceQueryEngine.
    '''

    def __init__(self, voxsel, space='voxel_indices', add_fa=None):
        '''Makes a new SurfaceVerticesQueryEngine

        Parameters
        ----------
        voxsel: volume_mask_dict.VolumeMaskDictionary
            mapping from center node indices to indices of voxels
            in a searchlight
        space: str (default: 'voxel_indices')
            defines by which space voxels are indexed.
        add_fa: list of str
            additional feature attributes that should be returned
            when this instance is called with a center node id.
        '''
        super(SurfaceVerticesQueryEngine, self).__init__()
        self.voxsel = voxsel
        self.space = space
        self._map_voxel_coord = None
        self._add_fa = add_fa

    def __repr__(self, prefixes=[]):
        return super(SurfaceVerticesQueryEngine, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['voxsel'])
            + _repr_attrs(self, ['space'], default='voxel_indices')
            + _repr_attrs(self, ['add_fa'], []))

    def __reduce__(self):
        return (self.__class__, (self.voxsel, self.space, self._add_fa),
                            dict(_map_voxel_coord=self._map_voxel_coord))

    def __str__(self):
        return '%s(%s, space=%s, add_fa=%s)' % (self.__class__.__name__,
                                                self.voxsel,
                                                self.space,
                                                self.add_fa)

    @property
    def ids(self):
        return self.voxsel.keys()

    def train(self, dataset):
        '''Train the query engine on a dataset'''
        vg = self.voxsel.volgeom
        # We are creating a map from big unmasked indices of voxels
        # known to voxsel into the dataset's feature indexes.
        # We verify that the current dataset has the necessary
        # features (i.e. are not masked out) and that the volume
        # geometry matches that of the original voxel selection

        vg_ds = None
        try:
            vg_ds = volgeom.from_any(dataset)
        except:
            vg_ds = None

        if vg_ds:
            eps = .0001
            if np.max(np.abs(vg_ds.affine - vg.affine)) > eps:
                raise ValueError("Mismatch in affine matrix: %r !+ %r" %
                                        (vg_ds.affine, vg.affine))
            if not vg_ds.same_shape(vg):
                raise ValueError("Mismatch in shape: (%s,%s,%s) != "
                                 "(%s,%s,%s)" %
                                        (vg_ds.shape[:3], vg.shape[:3]))
        else:
            warning("Could not find dataset volume geometry for %r" % dataset)


        self._map_voxel_coord = map_voxel_coord = {}
        long_is = vg.ijk2lin(dataset.fa[self.space].value)
        long_is_invol = vg.contains_lin(long_is)
        for i, long_i in enumerate(long_is):
            if not long_is_invol[i]:
                raise ValueError('Feature id %d (with voxel id %d)'
                                 ' is not in the (possibly masked) '
                                 'volume geometry %r)' % (i, long_i, vg))
            if long_i in map_voxel_coord:
                map_voxel_coord[long_i].append(i)
            else:
                map_voxel_coord[long_i] = [i]


    def untrain(self):
        self._map_voxel_coord = None

    def query_byid(self, vertexid):
        """Given a vertex ID give us indices of dataset features (voxels)

        Parameters
        ----------
        vertexid: int
            Index of searchlight center vertex on the surface.
            This value should be an element in self.ids

        Returns
        -------
        voxel_ids: list of int or AttrDataset
            The linear indices of voxels near the vertex with index vertexid.
            If the instance was constructed with add_fa=None, then voxel_ids
            is a list; otherwise it is a AttrDataset with additional feature
            attributes stored in voxel_ids.fa.

        """
        if self._map_voxel_coord is None:
            raise ValueError("No voxel mapping - did you train?")

        voxel_unmasked_ids = self.voxsel.get(vertexid)

        # map into dataset
        voxel_dataset_ids = [self._map_voxel_coord[i]
                             for i in voxel_unmasked_ids]
        voxel_dataset_ids_flat = sum(voxel_dataset_ids, [])

        if self._add_fa is not None:
            # optionally add additional information from voxsel
            ds = AttrDataset(np.asarray(voxel_dataset_ids_flat)[np.newaxis])
            for n in self._add_fa:
                fa_values = self.voxsel.get_aux(vertexid, n)
                assert(len(fa_values) == len(voxel_dataset_ids))
                ds.fa[n] = sum([[x] * len(ids)
                                for x, ids in zip(fa_values,
                                                  voxel_dataset_ids)], [])
            return ds
        return voxel_dataset_ids_flat



    def query(self, **kwargs):
        raise NotImplementedError


    def get_masked_nifti_image(self, center_ids=None):
        '''Return a NIfTI image binary mask with voxels covered by searchlights

        Parameters
        ----------
        center_ids: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all center_ids are used.

        Returns
        -------
        img: nibabel.Nifti1Image
            Nifti image with value zero for voxels that we not selected, and
            non-zero values for selected voxels.

        Notes
        -----
        When using surface-based searchlights, a use case of this function is
        to get the voxels that were associated with the searchlights in a
        subset of all nodes on a cortical surface.

        '''
        msk = self.voxsel.get_mask(keys=center_ids)
        import nibabel as nb
        img = nb.Nifti1Image(msk, self.voxsel.volgeom.affine)
        return img

    def linear_voxel_id2feature_id(self, linear_voxel_id):
        if type(linear_voxel_id) in (list, tuple):
            return map(self.linear_voxel_id2feature_id, linear_voxel_id)

        return self._map_voxel_coord[linear_voxel_id]

    def feature_id2linear_voxel_ids(self, feature_id):
        if type(feature_id) in (list, tuple):
            return map(self.feature_id2linear_voxel_ids, feature_id)

        return [i for i, j in self._map_voxel_coord.iteritems()
                                  if feature_id in j]

    def feature_id2nearest_vertex_id(self, feature_id,
                                     fallback_euclidean_distance=False):
        '''Compute the index of the vertex nearest to a given voxel.

        Parameters
        ----------
        feature_id: int
            Feature index (referring to a voxel).
        fallback_euclidean_distance: bool (default: False)
            If the voxel indexed by feature_id was not selected by any searchlight,
            then None is returned if fallback_euclidean_distance is False, but
            vertex_id with the nearest Euclidean distance is returned if True.

        Returns
        -------
        vertex_id: int
            Vertex index of vertex nearest to the feature with id feature_id.
            By default this function only considers vertices that are in one
            or more searchlights

        '''

        if type(feature_id) in (list, tuple):
            return map(self.feature_id2nearest_vertex_id, feature_id)

        lin_voxs = self.feature_id2linear_voxel_ids(feature_id)

        return self.voxsel.target2nearest_source(lin_voxs,
                      fallback_euclidean_distance=fallback_euclidean_distance)

    def vertex_id2nearest_feature_id(self, vertex_id):
        '''Compute the index of the voxel nearest to a given vertex.

        Parameters
        ----------
        vertex_id: int
            Vertex id (referring to a node on the surface).

        Returns
        -------
        feature_id: int
            Index of feature nearest to the vertex with id vertex_id.

        Notes
        -----
        This function only considers feature ids that are selected by
        at least one vertex_id..
        '''
        if type(vertex_id) in (list, tuple):
            return map(self.vertex_id2nearest_feature_id, vertex_id)

        lin_vox = self.voxsel.source2nearest_target(vertex_id)

        return self.linear_voxel_id2feature_id(lin_vox)

    def _set_add_fa(self, add_fa):
        if add_fa is not None:
            if not set(self.voxsel.aux_keys()).issuperset(add_fa):
                raise ValueError(
                    "add_fa should list only those known to voxsel %s"
                    % self.voxsel)
        self._add_fa = add_fa

    add_fa = property(fget=lambda self:self._add_fa, fset=_set_add_fa)


class SurfaceVoxelsQueryEngine(SurfaceVerticesQueryEngine):
    '''
    Query-engine that maps center voxels (indexed by feature ids)
    to indices of features (voxels) that are near each center voxel.

    In a typical use case such an instance is generated using
    the function 'disc_surface_queryengine' with the output_space='voxels'
    argument.

    For a mapping from center nodes (on a surface) to voxels,
    consider SurfaceVerticesQueryEngine.
    '''
    def __init__(self, voxsel, space='voxel_indices', add_fa=None,
                 fallback_euclidean_distance=True):
        '''Makes a new SurfaceVoxelsQueryEngine

        Parameters
        ----------
        voxsel: volume_mask_dict.VolumeMaskDictionary
            mapping from center node indices to indices of voxels
            in a searchlight
        space: str (default: 'voxel_indices')
            defines by which space voxels are indexed.
        add_fa: list of str
            additional feature attributes that should be returned
            when this instance is called with a center node id.
        fallback_euclidean_distance: bool (default: True)
            If True then every feature id will have voxels associated with
            it. That means that the number of self.ids is then equal to the
            number of features as the input dataset.
            If False, only feature ids that are selected by at least one
            searchlight are used. The number of self.ids is then equal
            to the number of voxels that are selected by at least one
            searchlight.
        '''
        super(SurfaceVoxelsQueryEngine, self).__init__(voxsel=voxsel,
                                                       space=space,
                                                       add_fa=add_fa)

        self._feature_id2vertex_id = None
        self.fallback_euclidean_distance = fallback_euclidean_distance


    def __repr__(self, prefixes=[]):
        prefixes_ = prefixes + _repr_attrs(self,
                                          ['fallback_euclidean_distance'],
                                          default=False)
        return super(SurfaceVoxelsQueryEngine, self).__repr__(
                            prefixes=prefixes_)

    def __reduce__(self):
        return (self.__class__, (self.voxsel, self.space,
                                 self._add_fa,
                                 self.fallback_euclidean_distance),
                                dict(_feature_id2vertex_id=self._feature_id2vertex_id))

    @property
    def ids(self):
        if self._feature_id2vertex_id is None:
            raise ValueError("No feature id mapping. Did you train?")
        return self._feature_id2vertex_id.keys()

    def query_byid(self, feature_id):
        '''Query the engine using a feature id'''
        vertex_id = self._feature_id2vertex_id[feature_id]
        return super(SurfaceVoxelsQueryEngine, self).query_byid(vertex_id)

    def train(self, ds):
        '''Train the query engine on a dataset'''

        super(SurfaceVoxelsQueryEngine, self).train(ds)

        # Compute the mapping from voxel (feature) ids to node ids

        fallback = self.fallback_euclidean_distance
        if fallback:
            # can use any feature id in ds
            feature_ids = range(ds.nfeatures)
        else:
            # see which feature ids were mapped to
            feature_ids = set()
            for v in self._map_voxel_coord.itervalues():
                feature_ids.update(set(v))

        f = lambda x:self.feature_id2nearest_vertex_id(x, fallback)

        fv = [(fid, f(fid)) for fid in feature_ids]

        # in the case of not fallback, some feature ids do not map to
        # a voxel id (i.e. they are mapped to None). Remove those from the
        # output
        self._feature_id2vertex_id = dict((f, v) for f, v in fv
                                                if not v is None)


    def untrain(self):
        super(SurfaceVoxelsQueryEngine, self).untrain(ds)
        self._feature_id2vertex_id = None

    def get_masked_nifti_image(self, center_ids=None):
        '''
        Returns a nifti image indicating which voxels are included
        in one or more searchlights.

        Parameters
        ----------
        center_ids: list or None
            Indices of center ids for which the associated masks must be
            used. If None, all center_ids are used.

        Returns
        -------
        img: nibabel.Nifti1Image
            Nifti image with value zero for voxels that we not selected, and
            non-zero values for selected voxels.

        Notes
        -----
        When using surface-based searchlights, a use case of this function is
        to get the voxels that were associated with the searchlights in a
        subset of all nodes on a cortical surface.
        '''
        if center_ids is None:
            center_ids = self.ids

        vertex_ids = [self._feature_id2vertex_id[center_id]
                                for center_id in center_ids]
        parent = super(SurfaceVoxelsQueryEngine, self)
        return parent.get_masked_nifti_image(center_ids=vertex_ids)




def disc_surface_queryengine(radius, volume, white_surf, pial_surf,
                             source_surf=None, source_surf_nodes=None,
                             volume_mask=False, distance_metric='dijkstra',
                             start_mm=0, stop_mm=0, start_fr=0., stop_fr=1.,
                             nsteps=10, eta_step=1, add_fa=None, nproc=None,
                             outside_node_margin=None,
                             results_backend=None,
                             tmp_prefix='tmpvoxsel',
                             output_modality='surface',
                             node_voxel_mapping='maximal'):
    """
    Voxel selection wrapper for multiple center nodes on the surface

    WiP
    XXX currently the last parameter 'output_modality' determines
    what kind of query engine is returned - is that bad?

    XXX: have to decide whether to use minimal_voxel_mapping=True as default

    Parameters
    ----------
    radius: int or float
        Size of searchlight. If an integer, then it indicates the number of
        voxels. If a float, then it indicates the radius of the disc
    volume: Dataset or NiftiImage or volgeom.Volgeom
        Volume in which voxels are selected.
    white_surf: str of surf.Surface
        Surface of white-matter to grey-matter boundary, or filename
        of file containing such a surface.
    pial_surf: str of surf.Surface
        Surface of grey-matter to pial-matter boundary, or filename
        of file containing such a surface.
    source_surf: surf.Surface or None
        Surface used to compute distance between nodes. If omitted, it is
        the average of the gray and white surfaces.
    source_surf_nodes: list of int or numpy array or None
        Indices of nodes in source_surf that serve as searchlight center.
        By default every node serves as a searchlight center.
    volume_mask: None (default) or False or int
        Mask from volume to apply from voxel selection results. By default
        no mask is applied. If volume_mask is an integer k, then the k-th
        volume from volume is used to mask the data. If volume is a Dataset
        and has a property volume.fa.voxel_indices, then these indices
        are used to mask the data, unless volume_mask is False or an integer.
    distance_metric: str
        Distance metric between nodes. 'euclidean' or 'dijksta' (default)
    start_fr: float (default: 0)
            Relative start position of line in gray matter, 0.=white
            surface, 1.=pial surface
    stop_fr: float (default: 1)
        Relative stop position of line (as in start_fr)
    start_mm: float (default: 0)
        Absolute start position offset (as in start_fr)
    stop_mm: float (default: 0)
        Absolute start position offset (as in start_fr)
    nsteps: int (default: 10)
        Number of steps from white to pial surface
    eta_step: int (default: 1)
        After how many searchlights an estimate should be printed of the
        remaining time until completion of all searchlights
    add_fa: None or list of strings
        Feature attributes from a dataset that should be returned if the
        queryengine is called with a dataset.
    nproc: int or None
        Number of parallel threads. None means as many threads as the
        system supports. The pprocess is required for parallel threads; if
        it cannot be used, then a single thread is used.
    outside_node_margin: float or None (default)
        By default nodes outside the volume are skipped; using this
        parameters allows for a marign. If this value is a float (possibly
        np.inf), then all nodes within outside_node_margin Dijkstra
        distance from any node within the volume are still assigned
        associated voxels. If outside_node_margin is True, then a node is
        always assigned voxels regardless of its position in the volume.
    results_backend : 'native' or 'hdf5' or None (default).
        Specifies the way results are provided back from a processing block
        in case of nproc > 1. 'native' is pickling/unpickling of results by
        pprocess, while 'hdf5' would use h5save/h5load functionality.
        'hdf5' might be more time and memory efficient in some cases.
        If None, then 'hdf5' if used if available, else 'native'.
    tmp_prefix : str, optional
        If specified -- serves as a prefix for temporary files storage
        if results_backend == 'hdf5'.  Thus can specify the directory to use
        (trailing file path separator is not added automagically).
    output_modality: 'surface' or 'volume' (default: 'surface')
        Indicates whether the output is surface-based
    node_voxel_mapping: 'minimal' or 'maximal'
        If 'minimal' then each voxel is associated with at most one node.
        If 'maximal' it is associated with as many nodes that contain the
        voxel (default: 'maximal')

    Returns
    -------
    qe: SurfaceVerticesQueryEngine
        Query-engine that maps center nodes to indices of features
        (voxels) that are near each center node.
        If output_modality=='volume' then qe is of type subclass
        SurfaceVoxelsQueryEngine.
    """

    modality2class = dict(surface=SurfaceVerticesQueryEngine,
                        volume=SurfaceVoxelsQueryEngine)

    if not output_modality in modality2class:
        raise KeyError("Illegal modality %s: should be in %s" %
                            (output_modality, modality2class.keys()))

    voxsel = surf_voxel_selection.run_voxel_selection(
                                radius=radius, volume=volume,
                                white_surf=white_surf, pial_surf=pial_surf,
                                source_surf=source_surf,
                                source_surf_nodes=source_surf_nodes,
                                volume_mask=volume_mask,
                                distance_metric=distance_metric,
                                start_fr=start_fr, stop_fr=stop_fr,
                                start_mm=start_mm, stop_mm=stop_mm,
                                nsteps=nsteps, eta_step=eta_step, nproc=nproc,
                                outside_node_margin=outside_node_margin,
                                results_backend=results_backend,
                                tmp_prefix=tmp_prefix,
                                node_voxel_mapping=node_voxel_mapping)


    qe = modality2class[output_modality](voxsel, add_fa=add_fa)

    return qe
