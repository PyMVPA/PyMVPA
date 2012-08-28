# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""QueryEngine for querying feature ids based on the surface nodes"""


__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.dataset import AttrDataset
from mvpa2.misc.neighborhood import QueryEngineInterface
if __debug__:
    from mvpa2.base import debug

class SurfaceVerticesQueryEngine(QueryEngineInterface):
    """TODO

    """
    def __init__(self, voxsel, space='voxel_indices', add_fa=None):
        super(SurfaceVerticesQueryEngine, self).__init__()
        self.voxsel = voxsel
        self.space = space
        self._map_voxel_coord = None

        if add_fa is not None:
            if not set(voxsel.sa_labels()).issuperset(add_fa):
                raise ValueError(
                    "add_fa should list only those known to voxsel %s"
                    % voxsel)
        self._add_fa = add_fa


    def __len__(self):
        # TODO : here it is a copy -- make it faster avoiding copies
        return len(self.voxsel.keys())


    def train(self, dataset):
        vg = self.voxsel.volgeom
        # We are creating a map from big unmasked indices of voxels
        # known to voxsel into the dataset's feature indexes
        # TODO:  this would fail in case of spatio-temporal analysis
        #        we would need to have a map from long vertex id to
        #        lists of dataset feature ids
        self._map_voxel_coord = dict(
            zip(vg.ijk2lin(dataset.fa[self.space].value),
                xrange(dataset.nfeatures)))

    def untrain(self):
        self._map_voxel_coord = None


    def query_byid(self, vertexid):
        """Given a vertex ID give us indices of dataset features (voxels)
        """
        # TODO: make lin_vox_idxs a parameter
        voxel_unmasked_ids = self.voxsel.get(vertexid, 'lin_vox_idxs')
        # map into dataset
        voxel_dataset_ids = [self._map_voxel_coord[i]
                             for i in voxel_unmasked_ids]
        if self._add_fa is not None:
            # optionally add additional information from voxsel
            ds = AttrDataset(np.asarray(voxel_dataset_ids)[np.newaxis])
            for n in self._add_fa:
                ds.fa[n] = self.voxsel.get(vertexid, n)
            return ds
        return voxel_dataset_ids


    def query(self, **kwargs):
        raise NotImplemented
