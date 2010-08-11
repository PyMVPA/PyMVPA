# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generic wrappers for learners (classifiers) provided by scikits.learn"""

__docformat__ = 'restructuredtext'

from mvpa.base import warning, externals
from mvpa.clfs.base import Classifier, accepts_dataset_as_samples, \
     FailedToTrainError, FailedToPredictError, DegenerateInputError


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
    
    """

    __tags__ = ['skl']

    def __init__(self, skl_learner, tags=None, **kwargs):
        """
        Parameters
        ----------
        skl_learner
          Existing instance of a learner from skl.  It should
          implement `fit` and `predict`.  If `predict_proba` is
          available in the interface, then conditional attribute
          `predict_proba` becomes available as well
        tags : list of string
          What additional tags to attach to this classifier.  Tags are
          used in the queries to classifier or regression warehouses.
        """

        self._skl_learner = skl_learner
        if tags:
            # So we make a per-instance copy
            self.__tags__ = self.__tags__ + tags
        Classifier.__init__(self, **kwargs)


    def __repr__(self):
        """String representation of `SKLLearnerWrapper`
        """
        return Classifier.__repr__(self, prefixes=[repr(self._skl_learner)])


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
        except ValueError, e:
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

        # Estimate estimates after predict, so if something goes
        # wrong, above exception handling occurs
        if self.ca.is_enabled('estimates'):
            if hasattr(self._skl_learner, 'predict_proba'):
                # Duplication of computation, since in many scenarios
                # predict() calls predict_proba()
                self.ca.estimates = self._skl_learner.predict_proba(data)
            else:
                warning("%s has no predict_proba() defined, so no probability"
                        " estimates could be extracted" % self._skl_learner)

        return res
