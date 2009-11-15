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
    _clf_internals = [ 'gnb', 'linear', 'nonlinear' ]

    common_variance = Parameter(False, allowedtype='bool',
             doc="""Assume variance common across all classes.""")

    def __init__(self, **kwargs):
        """Initialize an GNB classifier.
        """

        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint friendly initializations
        self.means = None
        self.stds = None


    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """
        # get the dataset information into easy vars
        X = dataset.samples
        labels = dataset.labels
        ulabels = dataset.uniquelabels
        #params = self.params        # for quicker access

        common_variance = self.params.common_variance

        # set the feature dimensions
        ns, nd = X.shape

        # XXX Sloppy implementation for now but it would have its advantages:
        #     0.  simple and straightforward
        #     1.  no data copying
        #     2.  uses dictionaries, so no need to care about
        #         mapping labels back and forth
        s_shape = X.shape[1:]           # shape of a single sample
        means = dict([(l, {}) for l in ulabels])
        stds = dict([(l, {}) for l in ulabels])
        nsamples = dict([(l, 0) for l in ulabels])

        # Estimate means and number of samples per each label
        for s, l in zip(X, labels):
            XXX

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "training finished on data.shape=%s " %
                  (X.shape, ) +
                  "min:max(data)=%f:%f, got min:max(w)=%f:%f" %
                  (N.min(X), N.max(X), N.min(b_mean), N.max(b_mean)))

    def untrain(self):
        """Untrain classifier and reset all learnt params
        """
        self.means = None
        self.stds = None
        self._class_labels = None
        super(GNB, self).reset()


    def _predict(self, data):
        """Predict the output for the provided data.
        """


        # XXX
        self.values = None              # set to the probabilities

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "predict on data.shape=%s min:max(data)=%f:%f " %
                  (`data.shape`, N.min(data), N.max(data)))

        return dot_prod


    # XXX Later come up with some
    #     could be a simple t-test maps using distributions
    #     per each class
    #def getSensitivityAnalyzer(self, **kwargs):
    #    """Returns a sensitivity analyzer for GNB."""
    #    return GNBWeights(self, **kwargs)


    # XXX Is there any reason to use properties?
    #means = property(lambda self: self.__biases)
    #stds = property(lambda self: self.__weights)



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

