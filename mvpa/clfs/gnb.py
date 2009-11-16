# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Gaussian Naive Bayes Classifier

   EXPERIMENTAL ;)
   Basic implementation of Gaussian Naive Bayes classifier.
"""

__docformat__ = 'restructuredtext'

import numpy as N

from numpy import ones, zeros, sum, abs, isfinite, dot
from mvpa.base import warning, externals
from mvpa.clfs.base import Classifier
from mvpa.misc.param import Parameter
from mvpa.misc.state import StateVariable
#from mvpa.measures.base import Sensitivity
#from mvpa.misc.transformers import SecondAxisMaxOfAbs # XXX ?


if __debug__:
    from mvpa.base import debug


class GNB(Classifier):
    """Gaussian Naive Bayes `Classifier`.

    """

    # Has linear separation iff
    # 1. number of training samples is balanced across classes
    # and
    # 2. variance is told to be class-independent

    # XXX decide when should we set corresponding internal,
    #     since it depends actually on the data -- no clear way,
    #     so set both linear and non-linear
    _clf_internals = [ 'gnb', 'linear', 'non-linear' ]

    common_variance = Parameter(False, allowedtype='bool',
             doc="""Use the same variance across all classes.""")
    uniform_prior = Parameter(True, allowedtype='bool',
             doc="""Use the same prior for all classes.""")

    logprob = Parameter(True, allowedtype='bool',
             doc="""Operate on log probabilities.  Preferable to avoid unneeded
             exponentiation and loose precision.
             If set, logprobs are stored in `values`""")
    normalize = Parameter(False, allowedtype='bool',
             doc="""Normalize (log)prob by P(data).  Requires probabilities thus for
             logprob case would require exponentiation of 'logprob's, thus
             disabled by default since does not impact classification per se.""")

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

    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """
        params = self.params

        # get the dataset information into easy vars
        X = dataset.samples
        labels = dataset.labels
        self.ulabels = ulabels = dataset.uniquelabels
        nlabels = len(ulabels)
        #params = self.params        # for quicker access
        label2index = dict((l, il) for il,l in enumerate(ulabels))

        # set the feature dimensions
        nsamples = len(X)

        # XXX Sloppy implementation for now but it would have its advantages:
        #     0.  simple and straightforward
        #     1.  no data copying
        #     2.  should work for any dimensionality of samples
        #  Let's later see how more efficient, more numpy-friendly
        #  approaches would perform
        s_shape = X.shape[1:]           # shape of a single sample

        self.means = means = \
                     N.zeros((nlabels, ) + s_shape)
        self.variances = variances = \
                     N.zeros((nlabels, ) + s_shape)
        # degenerate dimension are added for easy broadcasting later on
        nsamples_per_class = N.zeros((nlabels,) + (1,)*len(s_shape))

        # Estimate means and number of samples per each label
        for s, l in zip(X, labels):
            il = label2index[l]         # index of the label
            nsamples_per_class[il] += 1
            means[il] += s

        ## Actually compute the means
        non0labels = (nsamples_per_class.squeeze() != 0)
        means[non0labels] /= nsamples_per_class[non0labels]

        # Estimate variances
        # better loop than repmat! ;)
        for s, l in zip(X, labels):
            il = label2index[l]         # index of the label
            variances[il] += (s - means[il])**2

        ## Actually compute the variances
        if params.common_variance:
            # we need to get global std
            cvar = N.sum(variances, axis=0)/nsamples # sum across labels
            # broadcast the same variance across labels
            variances[:] = cvar
        else:
            variances[non0labels] /= nsamples_per_class[non0labels]

        # Store prior probabilities
        if params.uniform_prior:
            self.priors = N.ones((nlabels,))/nlabels
        else:
            # XXX check if valid... mvpa has (1+nsamples_per_class)/(nlabels+nsamples)
            self.priors = N.squeeze(nsamples_per_class) / float(nsamples)

        # Precompute and store weighting coefficient for Gaussian
        if params.logprob:
            # it would be added to exponent
            self._norm_weight = -0.5 * N.log(2*N.pi*variances)
        else:
            self._norm_weight = 1.0/N.sqrt(2*N.pi*variances)

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "training finished on data.shape=%s " %
                  (X.shape, ) +
                  "min:max(data)=%f:%f, got min:max(w)=%f:%f" %
                  (N.min(X), N.max(X), N.min(b_mean), N.max(b_mean)))


    def untrain(self):
        """Untrain classifier and reset all learnt params
        """
        self.means = None
        self.variances = None
        self.ulabels = None
        self.priors = None
        super(GNB, self).untrain()


    def _predict(self, data):
        """Predict the output for the provided data.
        """
        params = self.params
        # argument of exponentiation
        scaled_distances = \
            -0.5 * (((data - self.means[:, N.newaxis, ...])**2) \
                          / self.variances[:, N.newaxis, ...])
        if params.logprob:
            # if self.params.common_variance:
            # XXX YOH:
            # For decision there is no need to actually compute
            # properly scaled p, ie 1/sqrt(2pi * sigma_i) could be
            # simply discarded since it is common across features AND
            # classes
            # For completeness -- computing everything now even in logprob
            lprob_csfs = self._norm_weight[:, N.newaxis, ...] + scaled_distances

            # XXX for now just cut/paste with different operators, but
            #     could just bind them and reuse in the same equations
            # Naive part -- just a product of probabilities across features
            ## First we need to reshape to get class x samples x features
            lprob_csf = lprob_csfs.reshape(
                lprob_csfs.shape[:2] + (-1,))
            ## Now -- sum across features
            lprob_cs = lprob_csf.sum(axis=2)

            # Incorporate class probabilities:
            prob_cs_cp = lprob_cs + N.log(self.priors[:, N.newaxis])

        else:
            # Just a regular Normal distribution with per
            # feature/class mean and variances
            prob_csfs = \
                 self._norm_weight[:, N.newaxis, ...] * N.exp(scaled_distances)

            # Naive part -- just a product of probabilities across features
            ## First we need to reshape to get class x samples x features
            prob_csf = prob_csfs.reshape(
                prob_csfs.shape[:2] + (-1,))
            ## Now -- product across features
            prob_cs = prob_csf.prod(axis=2)

            # Incorporate class probabilities:
            prob_cs_cp = prob_cs * self.priors[:, N.newaxis]

        # Normalize by evidence P(data)
        if params.normalize:
            if params.logprob:
                prob_cs_cp_real = N.exp(prob_cs_cp)
            else:
                prob_cs_cp_real = prob_cs_cp
            prob_s_cp_marginals = N.sum(prob_cs_cp_real, axis=0)
            if params.logprob:
                prob_cs_cp -= N.log(prob_s_cp_marginals)
            else:
                prob_cs_cp /= prob_s_cp_marginals

        # Take the class with maximal (log)probability
        winners = prob_cs_cp.argmax(axis=0)
        predictions = [self.ulabels[c] for c in winners]


        self.values = prob_cs_cp.T         # set to the probabilities per class

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "predict on data.shape=%s min:max(data)=%f:%f " %
                  (`data.shape`, N.min(data), N.max(data)))

        return predictions


    # XXX Later come up with some
    #     could be a simple t-test maps using distributions
    #     per each class
    #def getSensitivityAnalyzer(self, **kwargs):
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

