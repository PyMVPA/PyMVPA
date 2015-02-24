# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Gaussian Naive Bayes Classifier

   Basic implementation of Gaussian Naive Bayes classifier.
"""

"""
TODO: for now all estimates are allocated at the first level of GNB
instance (e.g. self.priors, etc) -- move them deeper or into a
corresponding "Collection"?
The same for GNBSearchlight
"""

__docformat__ = 'restructuredtext'

import numpy as np

from numpy import ones, zeros, sum, abs, isfinite, dot
from mvpa2.base import warning, externals
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples
from mvpa2.base.param import Parameter
from mvpa2.base.state import ConditionalAttribute
from mvpa2.base.constraints import EnsureChoice
#from mvpa2.measures.base import Sensitivity


if __debug__:
    from mvpa2.base import debug

__all__ = [ "GNB" ]

class GNB(Classifier):
    """Gaussian Naive Bayes `Classifier`.

    `GNB` is a probabilistic classifier relying on Bayes rule to
    estimate posterior probabilities of labels given the data.  Naive
    assumption in it is an independence of the features, which allows
    to combine per-feature likelihoods by a simple product across
    likelihoods of "independent" features.
    See http://en.wikipedia.org/wiki/Naive_bayes for more information.

    Provided here implementation is "naive" on its own -- various
    aspects could be improved, but it has its own advantages:

    - implementation is simple and straightforward
    - no data copying while considering samples of specific class
    - provides alternative ways to assess prior distribution of the
      classes in the case of unbalanced sets of samples (see parameter
      `prior`)
    - makes use of NumPy broadcasting mechanism, so should be
      relatively efficient
    - should work for any dimensionality of samples

    `GNB` is listed both as linear and non-linear classifier, since
    specifics of separating boundary depends on the data and/or
    parameters: linear separation is achieved whenever samples are
    balanced (or ``prior='uniform'``) and features have the same
    variance across different classes (i.e. if
    ``common_variance=True`` to enforce this).

    Whenever decisions are made based on log-probabilities (parameter
    ``logprob=True``, which is the default), then conditional
    attribute `values`, if enabled, would also contain
    log-probabilities.  Also mention that normalization by the
    evidence (P(data)) is disabled by default since it has no impact
    per se on classification decision.  You might like to set
    parameter normalize to True if you want to access properly scaled
    probabilities in `values` conditional attribute.
    """
    # XXX decide when should we set corresponding internal,
    #     since it depends actually on the data -- no clear way,
    #     so set both linear and non-linear
    __tags__ = [ 'gnb', 'linear', 'non-linear',
                       'binary', 'multiclass', 'oneclass' ]

    common_variance = Parameter(False, constraints='bool',
             doc="""Use the same variance across all classes.""")

    prior = Parameter('laplacian_smoothing',
             constraints=EnsureChoice('laplacian_smoothing', 'uniform', 'ratio'),
             doc="""How to compute prior distribution.""")

    logprob = Parameter(True, constraints='bool',
             doc="""Operate on log probabilities.  Preferable to avoid unneeded
             exponentiation and loose precision.
             If set, logprobs are stored in `values`""")

    normalize = Parameter(False, constraints='bool',
             doc="""Normalize (log)prob by P(data).  Requires probabilities thus
             for `logprob` case would require exponentiation of 'logprob's, thus
             disabled by default since does not impact classification output.
             """)

    def __init__(self, **kwargs):
        """Initialize an GNB classifier.
        """

        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint friendly initializations
        self.means = None
        """Means of features per class"""
        self.variances = None
        """Variances per class, but "vars" is taken ;)"""
        self.ulabels = None
        """Labels classifier was trained on"""
        self.priors = None
        """Class probabilities"""

        # Define internal state of classifier
        self._norm_weight = None

    def _get_priors(self, nlabels, nsamples, nsamples_per_class):
        """Return prior probabilities given data
        """
        # helper function - squash all dimensions but 1
        squash = lambda x: np.atleast_1d(x.squeeze())

        prior = self.params.prior
        if prior == 'uniform':
            priors = np.ones((nlabels,))/nlabels
        elif prior == 'laplacian_smoothing':
            priors = (1+squash(nsamples_per_class)) \
                          / (float(nsamples) + nlabels)
        elif prior == 'ratio':
            priors = squash(nsamples_per_class) / float(nsamples)
        else:
            raise ValueError(
                "No idea on how to handle '%s' way to compute priors"
                % self.params.prior)
        return priors

    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """
        params = self.params
        targets_sa_name = self.get_space()
        targets_sa = dataset.sa[targets_sa_name]

        # get the dataset information into easy vars
        X = dataset.samples
        labels = targets_sa.value
        self.ulabels = ulabels = targets_sa.unique
        nlabels = len(ulabels)
        label2index = dict((l, il) for il, l in enumerate(ulabels))

        # set the feature dimensions
        nsamples = len(X)
        s_shape = X.shape[1:]           # shape of a single sample

        self.means = means = \
                     np.zeros((nlabels, ) + s_shape)
        self.variances = variances = \
                     np.zeros((nlabels, ) + s_shape)
        # degenerate dimension are added for easy broadcasting later on
        nsamples_per_class = np.zeros((nlabels,) + (1,)*len(s_shape))

        # Estimate means and number of samples per each label
        for s, l in zip(X, labels):
            il = label2index[l]         # index of the label
            nsamples_per_class[il] += 1
            means[il] += s

        # helper function - squash all dimensions but 1
        squash = lambda x: np.atleast_1d(x.squeeze())
        ## Actually compute the means
        non0labels = (squash(nsamples_per_class) != 0)
        means[non0labels] /= nsamples_per_class[non0labels]

        # Store prior probabilities
        self.priors = self._get_priors(nlabels, nsamples, nsamples_per_class)

        # Estimate variances
        # better loop than repmat! ;)
        for s, l in zip(X, labels):
            il = label2index[l]         # index of the label
            variances[il] += (s - means[il])**2

        ## Actually compute the variances
        if params.common_variance:
            # we need to get global std
            cvar = np.sum(variances, axis=0)/nsamples # sum across labels
            # broadcast the same variance across labels
            variances[:] = cvar
        else:
            variances[non0labels] /= nsamples_per_class[non0labels]

        # Precompute and store weighting coefficient for Gaussian
        if params.logprob:
            # it would be added to exponent
            self._norm_weight = -0.5 * np.log(2*np.pi*variances)
        else:
            self._norm_weight = 1.0/np.sqrt(2*np.pi*variances)

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "training finished on data.shape=%s " % (X.shape, )
                  + "min:max(data)=%f:%f" % (np.min(X), np.max(X)))


    def _untrain(self):
        """Untrain classifier and reset all learnt params
        """
        self.means = None
        self.variances = None
        self.ulabels = None
        self.priors = None
        super(GNB, self)._untrain()


    @accepts_dataset_as_samples
    def _predict(self, data):
        """Predict the output for the provided data.
        """
        params = self.params
        # argument of exponentiation
        scaled_distances = \
            -0.5 * (((data - self.means[:, np.newaxis, ...])**2) \
                          / self.variances[:, np.newaxis, ...])
        if params.logprob:
            # if self.params.common_variance:
            # XXX YOH:
            # For decision there is no need to actually compute
            # properly scaled p, ie 1/sqrt(2pi * sigma_i) could be
            # simply discarded since it is common across features AND
            # classes
            # For completeness -- computing everything now even in logprob
            lprob_csfs = self._norm_weight[:, np.newaxis, ...] \
                         + scaled_distances

            # XXX for now just cut/paste with different operators, but
            #     could just bind them and reuse in the same equations
            # Naive part -- just a product of probabilities across features
            ## First we need to reshape to get class x samples x features
            lprob_csf = lprob_csfs.reshape(
                lprob_csfs.shape[:2] + (-1,))
            ## Now -- sum across features
            lprob_cs = lprob_csf.sum(axis=2)

            # Incorporate class probabilities:
            prob_cs_cp = lprob_cs + np.log(self.priors[:, np.newaxis])

        else:
            # Just a regular Normal distribution with per
            # feature/class mean and variances
            prob_csfs = \
                 self._norm_weight[:, np.newaxis, ...] \
                 * np.exp(scaled_distances)

            # Naive part -- just a product of probabilities across features
            ## First we need to reshape to get class x samples x features
            prob_csf = prob_csfs.reshape(
                prob_csfs.shape[:2] + (-1,))
            ## Now -- product across features
            prob_cs = prob_csf.prod(axis=2)

            # Incorporate class probabilities:
            prob_cs_cp = prob_cs * self.priors[:, np.newaxis]

        # Normalize by evidence P(data)
        if params.normalize:
            if params.logprob:
                prob_cs_cp_real = np.exp(prob_cs_cp)
            else:
                prob_cs_cp_real = prob_cs_cp
            prob_s_cp_marginals = np.sum(prob_cs_cp_real, axis=0)
            if params.logprob:
                prob_cs_cp -= np.log(prob_s_cp_marginals)
            else:
                prob_cs_cp /= prob_s_cp_marginals

        # Take the class with maximal (log)probability
        winners = prob_cs_cp.argmax(axis=0)
        predictions = [self.ulabels[c] for c in winners]

        # set to the probabilities per class
        self.ca.estimates = prob_cs_cp.T

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "predict on data.shape=%s min:max(data)=%f:%f " %
                  (data.shape, np.min(data), np.max(data)))

        return predictions


    # XXX Later come up with some
    #     could be a simple t-test maps using distributions
    #     per each class
    #def get_sensitivity_analyzer(self, **kwargs):
    #    """Returns a sensitivity analyzer for GNB."""
    #    return GNBWeights(self, **kwargs)


    # XXX Is there any reason to use properties?
    #means = property(lambda self: self.__biases)
    #variances = property(lambda self: self.__weights)



## class GNBWeights(Sensitivity):
##     """`SensitivityAnalyzer` that reports the weights GNB trained
##     on a given `Dataset`.

##     """

##     _LEGAL_CLFS = [ GNB ]

##     def _call(self, dataset=None):
##         """Extract weights from GNB classifier.

##         GNB always has weights available, so nothing has to be computed here.
##         """
##         clf = self.clf
##         means = clf.means
##           XXX we can do something better ;)
##         return means

