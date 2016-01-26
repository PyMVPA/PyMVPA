# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Label adjacent features in a dataset
"""
import numpy as np

from ..base.param import Parameter
from ..datasets.base import Dataset

from .base import Measure

if __debug__:
    from ..base import debug

class Labeler(Measure):
    """Sensitivities of features for a given Classifier.

    """
    qe = Parameter(
            None,
            doc="QueryEngine to determine neighbors of any given feature")

    def __init__(self, qe, space='maxlabels', **kwargs):
        """Initialize labeler with a query engine

        Parameters
        ----------
        qe : `QueryEngine`
          Query engine to use.
        """
        super(Labeler, self).__init__(qe=qe, space=space, **kwargs)
        self._untrain()

    @property
    def is_trained(self):
        return self._nfeatures_trained is not None

    def _train(self, ds):
        self.params.qe.train(ds)
        self._nfeatures_trained = ds.nfeatures

    def _untrain(self):
        """Untrain QE
        """
        # they don't untrain apparently!! TODO
        # self.params.qe.untrain()
        self._nfeatures_trained = None
        super(Labeler, self)._untrain()

    def _call(self, ds):
        # few assertions so we don't support something what we don't support
        assert(ds.samples.ndim == 2)
        assert(ds.nfeatures == self._nfeatures_trained)

        need_64bits = ds.samples.size >= (2**31 - 2)
        lmaps = np.zeros(ds.shape, dtype=np.intp if need_64bits else np.int32)
        qe = self.params.qe
        qe_ids = set(qe.ids)
        maxlabels = []
        # we will support having multiple samples labeled independently
        for d, lmap in zip(ds.samples, lmaps):
            nonzero = np.nonzero(d)
            assert(len(nonzero) == 1)  # only 1 coordinate

            # Possibly a silly and slow implementation.  Will benchmark against scipy on 3d
            nonzero = set(nonzero[0])
            # we have indices of non-0 elements which constitute our initial seed points

            if __debug__:
                # for paranoid
                assert not nonzero.difference(qe_ids), "all of them must also be known to QE"

            idx = 0
            while nonzero:
                seed = nonzero.pop()
                if lmap[seed]:  # already labeled
                    continue
                idx += 1
                candidates = {seed}
                lmap[seed] = idx
                while candidates:
                    candidate = candidates.pop()
                    # process its neighbors
                    for neighbor in qe.query_byid(candidate):
                        if neighbor not in nonzero or lmap[neighbor]:
                            continue  # already labeled, so was considered etc
                            # or simply was 0 to start with
                        else:
                            # immediately label
                            lmap[neighbor] = idx
                            candidates.add(neighbor)
            maxlabels.append(idx)

        out = Dataset(lmaps, a=ds.a, sa=ds.sa, fa=ds.fa)
        out.sa[self.get_space()] = maxlabels
        return out
