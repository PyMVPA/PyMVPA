# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Gaussian Discriminant Analyses: LDA and QDA

   Basic implementation at the moment: no data sphering, nor
   dimensionality reduction tricks are in place ATM
"""

"""
TODO:

 * too much in common with GNB -- LDA/QDA/GNB could reuse much of machinery
 * provide actual probabilities computation as in GNB
 * LDA/QDA -- make use of data sphering and may be operating in the
              subspace of centroids

Was based on GNB code
"""

__docformat__ = 'restructuredtext'

import numpy as np

from numpy import ones, zeros, sum, abs, isfinite, dot
from mvpa2.base import warning, externals
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples
from mvpa2.base.learner import DegenerateInputError
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice
from mvpa2.base.state import ConditionalAttribute
#from mvpa2.measures.base import Sensitivity


if __debug__:
    from mvpa2.base import debug

__all__ = [ "LDA", "QDA" ]

class GDA(Classifier):
    """Gaussian Discriminant Analysis -- base for LDA and QDA

    """

    __tags__ = ['binary', 'multiclass', 'oneclass']


    prior = Parameter('laplacian_smoothing',
             constraints=EnsureChoice('laplacian_smoothing', 'uniform', 'ratio'),
             doc="""How to compute prior distribution.""")

    allow_pinv = Parameter(True,
             constraints='bool',
             doc="""Allow pseudo-inverse in case of degenerate covariance(s).""")

    def __init__(self, **kwargs):
        """Initialize a GDA classifier.
        """

        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint friendly initializations
        self.means = None
        """Means of features per class"""
        self.cov = None
        """Co-variances per class, but "vars" is taken ;)"""
        self.ulabels = None
        """Labels classifier was trained on"""
        self.priors = None
        """Class probabilities"""
        self.nsamples_per_class = None
        """Number of samples per class - used by derived classes"""

        # Define internal state of classifier
        self._norm_weight = None

    def _get_priors(self, nlabels, nsamples, nsamples_per_class):
        """Return prior probabilities given data
        """
        prior = self.params.prior
        if prior == 'uniform':
            priors = np.ones((nlabels,))/nlabels
        elif prior == 'laplacian_smoothing':
            priors = (1+np.squeeze(nsamples_per_class)) \
                          / (float(nsamples) + nlabels)
        elif prior == 'ratio':
            priors = np.squeeze(nsamples_per_class) / float(nsamples)
        else:
            raise ValueError, \
                  "No idea on how to handle '%s' way to compute priors" \
                  % self.params.prior
        return np.atleast_1d(priors)


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
        nfeatures = dataset.nfeatures

        self.means = means = \
                     np.zeros((nlabels, nfeatures))
        # degenerate dimension are added for easy broadcasting later on
        # XXX might want to remove -- for now taken from GNB as is
        self.nsamples_per_class = nsamples_per_class \
                                  = np.zeros((nlabels, 1))
        self.cov = cov = \
                     np.zeros((nlabels, nfeatures, nfeatures))


        # Estimate cov
        # better loop than repmat! ;)
        for l, il in label2index.iteritems():
            Xl = X[labels == l]
            nsamples_per_class[il] = len(Xl)
            # TODO: degenerate case... no samples for known label for
            #       some reason?
            means[il] = np.mean(Xl, axis=0)
            # since we have means already lets do manually cov here
            Xldm = Xl - means[il]
            cov[il] = np.dot(Xldm.T, Xldm)
            # scaling will be done correspondingly in LDA or QDA

        # Store prior probabilities
        self.priors = self._get_priors(nlabels, nsamples, nsamples_per_class)

        if __debug__ and 'GDA' in debug.active:
            debug('GDA', "training finished on data.shape=%s " % (X.shape, )
                  + "min:max(data)=%f:%f" % (np.min(X), np.max(X)))


    def _untrain(self):
        """Untrain classifier and reset all learnt params
        """
        self.means = None
        self.cov = None
        self.ulabels = None
        self.priors = None
        super(GDA, self)._untrain()


    @accepts_dataset_as_samples
    def _predict(self, data):
        """Predict the output for the provided data.
        """
        params = self.params

        self.ca.estimates = prob_cs_cp = self._g_k(data)

        # Take the class with maximal (log)probability
        # XXX in GNB it is axis=0, i.e. classes were first
        winners = prob_cs_cp.argmax(axis=1)
        predictions = [self.ulabels[c] for c in winners]

        if __debug__ and 'GDA' in debug.active:
            debug('GDA', "predict on data.shape=%s min:max(data)=%f:%f " %
                  (data.shape, np.min(data), np.max(data)))

        return predictions

    def _inv(self, cov):
        try:
            return np.linalg.inv(cov)
        except Exception, e:
            if self.params.allow_pinv:
                try:
                    return np.linalg.pinv(cov)
                except Exception, e:
                    pass
            raise DegenerateInputError, \
              "Data is probably singular, since inverse fails. Got %s"\
              % (e,)


class LDA(GDA):
    """Linear Discriminant Analysis.
    """

    __tags__ = GDA.__tags__ + ['linear', 'lda']


    def _untrain(self):
        self._w = None
        self._b = None
        super(LDA, self)._untrain()


    def _train(self, dataset):
        super(LDA, self)._train(dataset)
        nlabels = len(self.ulabels)
        # Sum and scale the covariance
        self.cov = cov = \
            np.sum(self.cov, axis=0) \
            / (np.sum(self.nsamples_per_class) - nlabels)

        # For now as simple as that -- see notes on top
        covi = self._inv(cov)

        # Precompute and store the actual separating hyperplane and offset
        self._w = np.dot(covi, self.means.T)
        self._b = b = np.zeros((nlabels,))
        for il in xrange(nlabels):
            m = self.means[il]
            b[il] = np.log(self.priors[il]) - 0.5 * np.dot(np.dot(m.T, covi), m)

    def _g_k(self, data):
        """Return decision function values"""
        return np.dot(data, self._w) + self._b


class QDA(GDA):
    """Quadratic Discriminant Analysis.
    """

    __tags__ = GDA.__tags__ + ['non-linear', 'qda']

    def _untrain(self):
        # XXX theoretically we could use the same _w although with
        # different "content"
        self._icov = None
        self._b = None
        super(QDA, self)._untrain()

    def _train(self, dataset):
        super(QDA, self)._train(dataset)

        # XXX should we drag cov around at all then?
        self._icov = np.zeros(self.cov.shape)

        for ic, cov in enumerate(self.cov):
            cov /= float(self.nsamples_per_class[ic])
            self._icov[ic] = self._inv(cov)

        self._b = np.array([np.log(p) - 0.5 * np.log(np.linalg.det(c))
                            for p,c in zip(self.priors, self.cov)])

    def _g_k(self, data):
        """Return decision function values"""
        res = []
        for m, covi, b in zip(self.means, self._icov, self._b):
            dm = data - m
            res.append(b - 0.5 * np.sum(np.dot(dm, covi) * dm, axis=1))
        return np.array(res).T
