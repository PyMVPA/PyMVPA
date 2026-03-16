# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Linear Discriminant Analysis with Ledoit-Wolf or OAS covariance shrinkage
"""

"""

Was based on GDA code
"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice, EnsureInt
from mvpa2.clfs.gda import GDA
from sklearn.covariance import LedoitWolf, OAS

if __debug__:
    from mvpa2.base import debug

__all__ = [ "ShrinkageLDA" ]

class ShrinkageLDA(GDA):
    """LDA using a shrinkage estimator for the covariance matrix
    Linear Discriminant Analysis classifier that estimates the within-class
    covariance matrix using a shrinkage estimator (Ledoit-Wolf)

    """

    __tags__ = GDA.__tags__ + ['linear', 'lda', 'shrinkage']

    prior = Parameter('laplacian_smoothing',
             constraints=EnsureChoice('laplacian_smoothing', 'uniform', 'ratio'),
             doc="""How to compute prior distribution.""")

    shrinkage_estimator = Parameter('ledoit-wolf',
            constraints=EnsureChoice('ledoit-wolf', 'oas'),
            doc="""Which shrinkage to use when estimating the covariance matrix.
            """)

    block_sz = Parameter(int(1000),
            constraints=EnsureInt(),
            doc="""Size of the blocks into which the covariance matrix will be
            split during its Ledoit-Wolf estimation. This is purely a memory
            optimization and does not affect results. (see scikit-learn doc)""")

    def __init__(self, **kwargs):
        super(ShrinkageLDA, self).__init__(**kwargs)

    def _train(self, dataset):

        """Train the classifier using `dataset` (`Dataset`).
        """

        # Adapted from GDA's method _train. Couldn't use that implementation
        # because it only computes the class-dependent SSCP matrices. The
        # shrinkage coefficient, however, needs to be computed from the whole
        # time series (with samples centered on their respective class means)
        params = self.params
        if params.shrinkage_estimator == 'ledoit-wolf':
            shrinkage_estimator = LedoitWolf(block_size=params.block_sz)
        elif params.shrinkage_estimator == 'oas':
            shrinkage_estimator = OAS()
        else:
            raise ValueError, "Unknown estimator '%s'" % \
            params.shrinkage_estimator

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

        # Subtract class means to compute within-class covariance later
        # better loop than repmat! ;)
        Xdm = list()
        for l, il in label2index.iteritems():
            Xl = X[labels == l]
            nsamples_per_class[il] = len(Xl)
            # TODO: degenerate case... no samples for known label for
            #       some reason?
            means[il] = np.mean(Xl, axis=0)
            # Instead of calculating within-class sum of squares, only subtract
            # the class mean here.
            Xdm.append(Xl - means[il])

        # Concatenate the centered time series
        Xdm = np.vstack(Xdm)

        # Store prior probabilities
        self.priors = self._get_priors(nlabels, nsamples, nsamples_per_class)

        if __debug__ and 'ShrinkageLDA' in debug.active:
            debug('ShrinkageLDA', "training finished on data.shape=%s " % (X.shape, )
                  + "min:max(data)=%f:%f" % (np.min(X), np.max(X)))

        nlabels = len(self.ulabels)

        # Use shrinkage estimator to compute covariance matrix
        self.cov = cov = shrinkage_estimator.fit(Xdm).covariance_

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
