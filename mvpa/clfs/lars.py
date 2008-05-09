#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Least angle regression (LARS) classifier."""

__docformat__ = 'restructuredtext'

# system imports
import numpy as N

import mvpa.base.externals as externals

if not externals.exists('rpy'):
    raise RuntimeError("Unable to load RPy which is needed for LARS.\n" +
                       "Please ensure that it was installed correctly.")

if not externals.exists('lars'):
    raise RuntimeError("Unable to load LARS library from R with RPy.\n" +
                       "Please ensure that LARS library was installed correctly.")

import rpy
rpy.r.library('lars')

# local imports
from mvpa.clfs.base import Classifier

known_models = ('lasso', 'stepwise', 'lar', 'forward.stagewise')

class LARS(Classifier):
    """Least angle regression (LARS) `Classifier`.

    LARS is the model selection algorithm from:

    Bradley Efron, Trevor Hastie, Iain Johnstone and Robert
    Tibshirani, Least Angle Regression Annals of Statistics (with
    discussion) (2004) 32(2), 407-499. A new method for variable
    subset selection, with the lasso and 'epsilon' forward stagewise
    methods as special cases.

    Similar to SMLR, it performs a feature selection while performing
    classification, but instead of starting with all features, it
    starts with none and adds them in, which is similar to boosting.

    This classifier behaves more like a ridge regression in that it
    returns prediction values and it treats the training labels as
    continuous.

    In the true nature of the PyMVPA framework, this algorithm is
    actually implemented in R by Trevor Hastie and wrapped via RPy.
    To make use of LARS, you must have R and RPy installed as well as
    the LARS contributed package. You can install the R and RPy with
    the following command on Debian-based machines:

    sudo aptitude install python-rpy python-rpy-doc r-base-dev

    You can then install the LARS package by running R as root and
    calling:

    install.packages()

    """

    _clf_internals = [ 'regression' ]

    def __init__(self, model_type="lasso", trace=False, normalize=True,
                 intercept=True, max_steps=None, use_Gram=False, **kwargs):
        """
        Initialize LARS.

        See the help in R for further details on the following parameters:

        :Parameters:
          model_type : string
            Type of LARS to run. Can be one of ('lasso', 'lar',
            'forward.stagewise', 'stepwise').
          trace : boolean
            Whether to print progress in R as it works.
          normalize : boolean
            Whether to normalize the L2 Norm.
          intercept : boolean
            Whether to add a non-penalized intercept to the model.
          max_steps : None or int
            If not None, specify the total number of iterations to run. Each
            iteration adds a feature, but leaving it none will add until
            convergence.
          use_Gram : boolean
            Whether to compute the Gram matrix (this should be false if you
            have more features than samples.)
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        if not model_type in known_models:
            raise ValueError('Unknown model %s for LARS is specified. Known' %
                             model_type + 'are %s' % `known_models`)

        # set up the params
        self.__type = model_type
        self.__normalize = normalize
        self.__intercept = intercept
        self.__trace = trace
        self.__max_steps = max_steps
        self.__use_Gram = use_Gram

        # pylint friendly initializations
        self.__weights = None
        """The beta weights for each feature."""
        self.__trained_model = None
        """The model object after training that will be used for
        predictions."""

        # It does not make sense to calculate a confusion matrix for a
        # regression
        self.states.enable('training_confusion', False)

    def __repr__(self):
        """String summary of the object
        """
        return """LARS(type=%s, normalize=%s, intercept=%s, trace=%s, max_steps=%s, use_Gram=%s, enabled_states=%s)""" % \
               (self.__type,
                self.__normalize,
                self.__intercept,
                self.__trace,
                self.__max_steps,
                self.__use_Gram,
                str(self.states.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """
        if self.__max_steps is None:
            # train without specifying max_steps
            self.__trained_model = rpy.r.lars(data.samples,
                                              data.labels[:,N.newaxis],
                                              type=self.__type,
                                              normalize=self.__normalize,
                                              intercept=self.__intercept,
                                              trace=self.__trace,
                                              use_Gram=self.__use_Gram)
        else:
            # train with specifying max_steps
            self.__trained_model = rpy.r.lars(data.samples,
                                              data.labels[:,N.newaxis],
                                              type=self.__type,
                                              normalize=self.__normalize,
                                              intercept=self.__intercept,
                                              trace=self.__trace,
                                              use_Gram=self.__use_Gram,
                                              max_steps=self.__max_steps)

        # set the weights to the final state
        self.__weights = self.__trained_model['beta'][-1,:]

    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict with the final state (i.e., the last step)
        res = rpy.r.predict_lars(self.__trained_model,
                                 data,
                                 mode='step',
                                 s=self.__trained_model['beta'].shape[0])

        fit = N.asarray(res['fit'])
        if len(fit.shape) == 0:
            # if we just got 1 sample with a scalar
            fit = fit.reshape( (1,) )
        return fit

    weights = property(lambda self: self.__weights)

