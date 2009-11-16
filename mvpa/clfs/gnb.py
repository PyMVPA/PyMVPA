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
             doc="""Assume variance common across all classes.""")

    full = Parameter(True, allowedtype='bool',
             doc="""Full computation of probabilities.
             Just for classification (if not interested in actual
             probabilities stored in values) with `common_variance`=True
             it is possible to avoid computation of exponents... XXX""")

    def __init__(self, **kwargs):
        """Initialize an GNB classifier.
        """

        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint friendly initializations
        self.means = None
        """Means of features per class"""
        self.std2s = None
        """Variances per class, but "vars" is taken ;)"""
        self.ulabels = None
        """Labels classifier was trained on"""
        self.class_prob = None
        """Class probabilities"""

    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """
        # get the dataset information into easy vars
        X = dataset.samples
        labels = dataset.labels
        self.ulabels = ulabels = dataset.uniquelabels
        nlabels = len(ulabels)
        #params = self.params        # for quicker access
        label2index = dict((l, il) for il,l in enumerate(ulabels))

        common_variance = self.params.common_variance

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
        self.std2s = std2s = \
                     N.zeros((nlabels, ) + s_shape)
        # degenerate dimension are added for easy broadcasting later on
        nsamples_per_class = N.zeros((nlabels,) + (1,)*len(s_shape))

        # Estimate means and number of samples per each label
        for s, l in zip(X, labels):
            il = label2index[l]         # index of the label
            nsamples_per_class[il] += 1
            means[il] += s

        ## Actually compute the means
        non0labels = nsamples_per_class != 0
        means[non0labels] /= nsamples_per_class[non0labels]

        # Estimate std2s
        # better loop than repmat! ;)
        for s, l in zip(X, labels):
            il = label2index[l]         # index of the label
            std2s[il] += (s - means[il])**2

        ## Actually compute the std2s
        if common_variance:
            # we need to get global std
            cvar = N.sum(std2s, axis=0)/nsamples # sum across labels
            # broadcast the same variance across labels
            std2s[:] = cvar
        else:
            std2s[non0labels] /= nsamples_per_class[non0labels]

        # Store class_probabilities
        self.class_prob = N.squeeze(nsamples_per_class) / float(nsamples)

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "training finished on data.shape=%s " %
                  (X.shape, ) +
                  "min:max(data)=%f:%f, got min:max(w)=%f:%f" %
                  (N.min(X), N.max(X), N.min(b_mean), N.max(b_mean)))


    def untrain(self):
        """Untrain classifier and reset all learnt params
        """
        self.means = None
        self.std2s = None
        self.ulabels = None
        self.class_prob = None
        super(GNB, self).untrain()


    def _predict(self, data):
        """Predict the output for the provided data.
        """
        if self.params.full:
            # Just a regular Normal distribution with per
            # feature/class mean and std2s
            prob_csfs = \
                 1.0/N.sqrt(2*N.pi*self.std2s[:, N.newaxis, ...]) * \
                 N.exp(-0.5 * (((data - self.means[:, N.newaxis, ...])**2)\
                               / self.std2s[:, N.newaxis, ...]))
        else:
            # if self.params.common_variance:
            # XXX YOH:
            # For decision there is no need to actually compute
            # properly scaled p, ie 1/sqrt(2pi * sigma_i) could be
            # simply discarded since it is common across features AND
            # classes
            raise NotImplemented, '"Optimized" GNB prediction is not here yet'

        # Naive part -- just a product of probabilities across features
        ## First we need to reshape to get class x samples x features
        prob_csf = prob_csfs.reshape(
            prob_csfs.shape[:2] + (-1,))
        ## Now -- product across features
        prob_cs = prob_csf.prod(axis=2)

        # Incorporate class probabilities:
        prob_cs_cp = prob_cs * self.class_prob[:, N.newaxis]

        # Take the class with maximal probability
        winners = prob_cs_cp.argmax(axis=0)
        predictions = [self.ulabels[c] for c in winners]

        self.values = prob_cs.T         # set to the probabilities per class

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
    #std2s = property(lambda self: self.__weights)



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

