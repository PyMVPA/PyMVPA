# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generic wrappers for learners (classifiers) provided by R's MASS

Highly experimental and ad-hoc -- primary use was to verify LDA/QDA
results, thus not included in the mvpa2.suite ATM.
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import warning, externals
from mvpa2.base.state import ConditionalAttribute
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples
from mvpa2.base.learner import FailedToTrainError, FailedToPredictError


# do conditional to be able to build module reference
if externals.exists('mass', raise_=True):
    import rpy2.robjects
    import rpy2.robjects.numpy2ri
    if hasattr(rpy2.robjects.numpy2ri,'activate'):
        rpy2.robjects.numpy2ri.activate()
    RRuntimeError = rpy2.robjects.rinterface.RRuntimeError
    r = rpy2.robjects.r
    r.library('MASS')
    from mvpa2.support.rpy2_addons import Rrx, Rrx2


class MASSLearnerAdapter(Classifier):
    """Generic adapter for instances of learners provided by R's MASS

    Provides basic adaptation of interface for classifiers from MASS
    library (e.g. QDA, LDA), by adapting interface.

    Examples
    --------
    >>> if externals.exists('mass'):
    ...    from mvpa2.testing.datasets import datasets
    ...    mass_qda = MASSLearnerAdapter('qda', tags=['non-linear', 'multiclass'], enable_ca=['posterior'])
    ...    mass_qda.train(datasets['uni2large'])
    ...    mass_qda.predict(datasets['uni2large']) # doctest: +SKIP
    """

    __tags__ = ['mass', 'rpy2']

    posterior = ConditionalAttribute(enabled=False,
        doc='Posterior probabilities if provided by classifier')

    def __init__(self, learner, kwargs=None, kwargs_predict=None,
                 tags=None, **kwargs_):
        """
        Parameters
        ----------
        learner : string
        kwargs : dict, optional
        kwargs_predict : dict, optional
        tags : list of string
          What additional tags to attach to this classifier.  Tags are
          used in the queries to classifier or regression warehouses.
        """

        self._learner = learner

        self._kwargs = kwargs or {}
        self._kwargs_predict = kwargs_predict or {}

        if tags:
            # So we make a per-instance copy
            self.__tags__ = self.__tags__ + tags

        Classifier.__init__(self, **kwargs_)


    def __repr__(self):
        """String representation of `SKLLearnerWrapper`
        """
        return Classifier.__repr__(self,
            prefixes=[repr(self._learner),
                      'kwargs=%r' % (self._kwargs,)])


    def _train(self, dataset):
        """Train the skl learner using `dataset` (`Dataset`).
        """
        targets_sa = dataset.sa[self.get_space()]
        targets = targets_sa.value
        if not 'regression' in self.__tags__:
            targets = self._attrmap.to_numeric(targets)

        try:
            self._R_model = r[self._learner](
                dataset.samples,
                targets,
                **self._kwargs)
        except RRuntimeError, e:
            raise FailedToTrainError, \
                  "Failed to train %s on %s. Got '%s' during call to fit()." \
                  % (self, dataset, e)


    @accepts_dataset_as_samples
    def _predict(self, data):
        """Predict using the trained MASS learner
        """
        try:
            output = r.predict(self._R_model,
                               data,
                               **self._kwargs_predict)
            # TODO: access everything computed, and assign to
            #       ca's: res.names
            classes = Rrx2(output, 'class')
            # TODO: move to helper function to be used generically
            if classes.rclass[0] == 'factor':
                classes = [int(classes.levels[i-1]) for i in classes]
            if 'posterior' in output.names:
                self.ca.posterior = np.asarray(Rrx2(output, 'posterior'))
            res = np.asarray(classes)
        except Exception, e:
            raise FailedToPredictError, \
                  "Failed to predict %s on data of shape %s. Got '%s' during" \
                  " call to predict()." % (self, data.shape, e)

        return res
