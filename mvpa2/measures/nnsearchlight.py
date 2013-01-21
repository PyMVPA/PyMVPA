# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""An efficient implementation of searchlight for M1NN.
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere

from mvpa2.clfs.distance import squared_euclidean_distance

from mvpa2.measures.adhocsearchlightbase import SimpleStatBaseSearchlight, \
     _STATS

if __debug__:
    from mvpa2.base import debug
    import time as time

__all__ = [ "M1NNSearchlight", 'sphere_m1nnsearchlight' ]

class M1NNSearchlight(SimpleStatBaseSearchlight):
    """Efficient implementation of Mean-Nearest-Neighbor `Searchlight`.

    """

    @borrowkwargs(SimpleStatBaseSearchlight, '__init__')
    def __init__(self, knn, generator, qe, **kwargs):
        """Initialize a M1NNSearchlight
        TODO -- example? or just kill altogether
                rethink providing knn sample vs specifying all parameters
                explicitly
        Parameters
        ----------
        knn : `kNN`
          Used to fetch space and dfx settings. TODO
        """
        # verify that desired features are supported
        if knn.dfx != squared_euclidean_distance:
            raise ValueError(
                "%s distance function is not yet supported by M1NNSearchlight"
                % (knn.dfx,))

        # init base class first
        SimpleStatBaseSearchlight.__init__(self, generator, qe, **kwargs)

        self._knn = knn
        self.__pl_train = self.__pl_test = None

    def __repr__(self, prefixes=[]):
        return super(M1NNSearchlight, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['knn'])
            )


    def _get_space(self):
        return self.knn.get_space()

    def _untrain(self):
        super(M1NNSearchlight, self)._untrain()
        self.__pl_train = self.__pl_test = None

    def _reserve_pl_stats_space(self, shape):
        # per each label: to be (re)computed within each loop split
        # Let's try to reuse the memory though
        self.__pl_train = _STATS()
        self.__pl_test = _STATS()
        for pl in (self.__pl_train, self.__pl_test):
            pl.sums = np.zeros(shape)
            pl.means = np.zeros(shape)
            # means of squares for stddev computation
            pl.sums2 = np.zeros(shape)
            pl.variances = np.zeros(shape)
            # degenerate dimension are added for easy broadcasting later on
            pl.nsamples = np.zeros(shape[:1] + (1,)*(len(shape)-1))


    def _sl_call_on_a_split(self,
                            split, X,
                            training_sis, testing_sis,
                            nroi_fids, roi_fids,
                            indexsum_fx,
                            labels_numeric,
                            ):
        """Call to M1NNSearchlight
        """
        # Local bindings
        knn = self.knn
        params = knn.params

        pl_train = self.__pl_train
        pl_test  = self.__pl_test

        training_nsamples, training_non0labels = \
            self._compute_pl_stats(training_sis, pl_train)

        testing_nsamples, testing_non0labels = \
            self._compute_pl_stats(testing_sis, pl_test)

        nlabels = len(pl_train.nsamples)

        assert(len(np.unique(labels_numeric)) == nlabels)
        assert(training_non0labels == slice(None)) # not sure/tested if we can handle this one
        assert(testing_non0labels == slice(None)) # not sure/tested if we can handle this one

        # squared distances between the means...

        # hm, but we need for each combination of labels
        # so we keep 0th dimension corresponding to test "samples/labels"
        diff_pl_pl = pl_test.means[:, None] - pl_train.means[None,:]
        diff_pl_pl2 = np.square(diff_pl_pl)

        # XXX OPT: is it worth may be reserving the space beforehand?
        diff_pl_pl2_sl = np.zeros(diff_pl_pl2.shape[:-1] + (nroi_fids,))
        indexsum_fx(diff_pl_pl2, roi_fids, out=diff_pl_pl2_sl)

        # predictions are just the labels with minimal distance
        predictions = np.argmin(diff_pl_pl2_sl, axis=1)

        return np.asanyarray(self._ulabels_numeric), predictions

    knn = property(fget=lambda self: self._knn)

@borrowkwargs(M1NNSearchlight, '__init__', exclude=['roi_ids', 'queryengine'])
def sphere_m1nnsearchlight(knn, generator, radius=1, center_ids=None,
                          space='voxel_indices', *args, **kwargs):
    """Creates a `M1NNSearchlight` to assess :term:`cross-validation`
    classification performance of M1NN on all possible spheres of a
    certain size within a dataset.

    The idea of taking advantage of naiveness of M1NN for the sake of
    quick searchlight-ing stems from Francisco Pereira (paper under
    review).

    Parameters
    ----------
    radius : float
      All features within this radius around the center will be part
      of a sphere.
    center_ids : list of int
      List of feature ids (not coordinates) the shall serve as sphere
      centers. By default all features will be used (it is passed
      roi_ids argument for Searchlight).
    space : str
      Name of a feature attribute of the input dataset that defines the spatial
      coordinates of all features.
    **kwargs
      In addition this class supports all keyword arguments of
      :class:`~mvpa2.measures.nnsearchlight.M1NNSearchlight`.

    Notes
    -----
    If any `BaseSearchlight` is used as `SensitivityAnalyzer` one has to make
    sure that the specified scalar `Measure` returns large
    (absolute) values for high sensitivities and small (absolute) values
    for low sensitivities. Especially when using error functions usually
    low values imply high performance and therefore high sensitivity.
    This would in turn result in sensitivity maps that have low
    (absolute) values indicating high sensitivities and this conflicts
    with the intended behavior of a `SensitivityAnalyzer`.
    """
    # build a matching query engine from the arguments
    kwa = {space: Sphere(radius)}
    qe = IndexQueryEngine(**kwa)
    # init the searchlight with the queryengine
    return M1NNSearchlight(knn, generator, qe,
                          roi_ids=center_ids, *args, **kwargs)
