# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""An efficient implementation of searchlight for GNB.
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere

from mvpa2.measures.adhocsearchlightbase import \
     SimpleStatBaseSearchlight, _STATS

if __debug__:
    from mvpa2.base import debug
    import time as time

__all__ = [ "GNBSearchlight", 'sphere_gnbsearchlight' ]

class GNBSearchlight(SimpleStatBaseSearchlight):
    """Efficient implementation of Gaussian Naive Bayes `Searchlight`.

    This implementation takes advantage that :class:`~mvpa2.clfs.gnb.GNB` is
    "naive" in its reliance on massive univariate conditional
    probabilities of each feature given a target class.  Plain
    :class:`~mvpa2.measures.searchlight.Searchlight` analysis approach
    asks for the same information over again and over again for
    the same feature in multiple "lights".  So it becomes possible to
    drastically cut running time of a Searchlight by pre-computing basic
    statistics necessary used by GNB beforehand and then doing their
    subselection for a given split/feature set.

    Kudos for the idea and showing that it indeed might be beneficial
    over generic Searchlight with GNB go to Francisco Pereira.
    """

    @borrowkwargs(SimpleStatBaseSearchlight, '__init__')
    def __init__(self, gnb, generator, qe, **kwargs):
        """Initialize a GNBSearchlight

        Parameters
        ----------
        gnb : `GNB`
          `GNB` classifier as the specification of what GNB parameters
          to use. Instance itself isn't used.
        """

        # init base class first
        SimpleStatBaseSearchlight.__init__(self, generator, qe, **kwargs)

        self._gnb = gnb
        self.__pl_train = None


    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(GNBSearchlight, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['gnb'])
            )


    def _get_space(self):
        return self.gnb.get_space()

    def _untrain(self):
        super(GNBSearchlight, self)._untrain()
        self.__pl_train = None

    def _reserve_pl_stats_space(self, shape):
        # per each label: to be (re)computed within each loop split
        # Let's try to reuse the memory though
        pl = self.__pl_train = _STATS()
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
        """Call to GNBSearchlight
        """
        # Local bindings
        gnb = self.gnb
        params = gnb.params

        pl = self.__pl_train # we want to reuse the same storage across
                             # splits

        training_nsamples, non0labels = \
            self._compute_pl_stats(training_sis, pl)

        nlabels = len(pl.nsamples)

        if params.common_variance:
            pl.variances[:] = \
                np.sum(pl.sums2 - pl.sums * pl.means, axis=0) \
                / training_nsamples
        else:
            pl.variances[non0labels] = \
                (pl.sums2 - pl.sums * pl.means)[non0labels] \
                / pl.nsamples[non0labels]

        # assign priors
        priors = gnb._get_priors(
            nlabels, training_nsamples, pl.nsamples)

        # proceed in a way we have in GNB code with logprob=True,
        # i.e. operating within the exponents -- should lead to some
        # performance advantage
        norm_weight = -0.5 * np.log(2*np.pi*pl.variances)
        # last added dimension would be for ROIs
        logpriors = np.log(priors[:, np.newaxis, np.newaxis])

        if __debug__:
            debug('SLC', "  'Training' is done")

        # Now it is time to "classify" our samples.
        # and for that we first need to compute corresponding
        # probabilities (or may be un
        data = X[split[1].samples[:, 0]]

        # argument of exponentiation
        scaled_distances = \
             -0.5 * (((data - pl.means[:, np.newaxis, ...])**2) \
                     / pl.variances[:, np.newaxis, ...])

        # incorporate the normalization from normals
        lprob_csfs = norm_weight[:, np.newaxis, ...] + scaled_distances

        ## First we need to reshape to get class x samples x features
        lprob_csf = lprob_csfs.reshape(lprob_csfs.shape[:2] + (-1,))

        ## Now we come to naive part which requires looping
        ## through all spheres
        if __debug__:
            debug('SLC', "  Doing 'Searchlight'")
        # resultant logprobs for each class x sample x roi
        lprob_cs_sl = np.zeros(lprob_csfs.shape[:2] + (nroi_fids,))
        indexsum_fx(lprob_csf, roi_fids, out=lprob_cs_sl)

        lprob_cs_sl += logpriors
        lprob_cs_cp_sl = lprob_cs_sl
        # for each of the ROIs take the class with maximal (log)probability
        predictions = lprob_cs_cp_sl.argmax(axis=0)
        # no need to map back [self.ulabels[c] for c in winners]
        #predictions = winners

        targets = labels_numeric[testing_sis]

        return targets, predictions

    gnb = property(fget=lambda self: self._gnb)

@borrowkwargs(GNBSearchlight, '__init__', exclude=['roi_ids', 'queryengine'])
def sphere_gnbsearchlight(gnb, generator, radius=1, center_ids=None,
                          space='voxel_indices', *args, **kwargs):
    """Creates a `GNBSearchlight` to assess :term:`cross-validation`
    classification performance of GNB on all possible spheres of a
    certain size within a dataset.

    The idea of taking advantage of naiveness of GNB for the sake of
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
      :class:`~mvpa2.measures.gnbsearchlight.GNBSearchlight`.

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
    return GNBSearchlight(gnb, generator, qe,
                          roi_ids=center_ids, *args, **kwargs)
