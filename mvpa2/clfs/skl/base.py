# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generic wrappers for learners (classifiers) provided by scikit-learn (AKA sklearn)"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import warning, externals
from mvpa2.base.dochelpers import _repr_attrs
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples
from mvpa2.base.learner import FailedToTrainError, FailedToPredictError, \
        DegenerateInputError


# do conditional to be able to build module reference
externals.exists('skl', raise_=True)


class SKLLearnerAdapter(Classifier):
    """Generic adapter for instances of learners provided by scikits.learn

    Provides basic adaptation of interface (e.g. train -> fit) and
    wraps the constructed instance of a learner from skl, so it looks
    like any other learner present within PyMVPA (so obtains all the
    conditional attributes defined at the base level of a
    `Classifier`)

    Examples
    --------

    TODO
    """

    __tags__ = ['skl']

    def __init__(self, skl_learner, tags=None, enforce_dim=None,
                 **kwargs):
        """
        Parameters
        ----------
        skl_learner
          Existing instance of a learner from skl.  It should
          implement `fit` and `predict`.  If `predict_proba` is
          available in the interface, then conditional attribute
          `probabilities` becomes available as well
        tags : list of string
          What additional tags to attach to this learner.  Tags are
          used in the queries to classifier or regression warehouses.
        enforce_dim : None or int, optional
          If not None, it would enforce given dimensionality for
          ``predict`` call, if all other trailing dimensions are
          degenerate.
        """

        self._skl_learner = skl_learner
        self.enforce_dim = enforce_dim
        if tags:
            # So we make a per-instance copy
            self.__tags__ = self.__tags__ + tags
        Classifier.__init__(self, **kwargs)


    def __repr__(self):
        """String representation of `SKLLearnerWrapper`
        """
        prefixes = [repr(self._skl_learner)]
        if self.__tags__ != ['skl']:
            prefixes += ['tags=%r' % [t for t in self.__tags__ if t != 'skl']]
        prefixes += _repr_attrs(self, ['enforce_dim'])
        return Classifier.__repr__(self, prefixes=prefixes)


    def _train(self, dataset):
        """Train the skl learner using `dataset` (`Dataset`).
        """
        targets_sa = dataset.sa[self.get_space()]
        targets = targets_sa.value
        # Some sanity checking so some classifiers such as LDA do not
        # puke meaningless exceptions
        if 'lda' in self.__tags__:
            if not dataset.nsamples > len(targets_sa.unique):
                raise DegenerateInputError, \
                      "LDA requires # of samples exceeding # of classes"

        # we better map into numeric labels if it is not a regression
        if not 'regression' in self.__tags__:
            targets = self._attrmap.to_numeric(targets)

        try:
            # train underlying learner
            self._skl_learner.fit(dataset.samples, targets)
        except (ValueError, np.linalg.LinAlgError), e:
            raise FailedToTrainError, \
                  "Failed to train %s on %s. Got '%s' during call to fit()." \
                  % (self, dataset, e)

    @accepts_dataset_as_samples
    def _predict(self, data):
        """Predict using the skl learner
        """
        try:
            res = self._skl_learner.predict(data)
        except Exception, e:
            raise FailedToPredictError, \
                  "Failed to predict %s on data of shape %s. Got '%s' during" \
                  " call to predict()." % (self, data.shape, e)

        if self.enforce_dim:
            res_dim = len(res.shape)
            if res_dim > self.enforce_dim:
                # would throw meaningful exception if not possible
                res = res.reshape(res.shape[:self.enforce_dim])
            elif res_dim < self.enforce_dim:
                # broadcast
                res = res.reshape(res.shape + (1,)* (self.enforce_dim - res_dim))
        # Estimate estimates after predict, so if something goes
        # wrong, above exception handling occurs
        if self.ca.is_enabled('probabilities'):
            if hasattr(self._skl_learner, 'predict_proba'):
                # Duplication of computation, since in many scenarios
                # predict() calls predict_proba()
                self.ca.probabilities = self._skl_learner.predict_proba(data)
            else:
                warning("%s has no predict_proba() defined, so no probability"
                        " estimates could be extracted" % self._skl_learner)
        self.ca.estimates = res
        return res
