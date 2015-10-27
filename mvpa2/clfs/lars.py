# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Least angle regression (LARS)."""

__docformat__ = 'restructuredtext'

# system imports
import numpy as np

import mvpa2.base.externals as externals
from mvpa2.support.due import due, Doi, BibTeX

# do conditional to be able to build module reference
if externals.exists('lars', raise_=True):
    import rpy2.robjects
    import rpy2.robjects.numpy2ri
    if hasattr(rpy2.robjects.numpy2ri,'activate'):
        rpy2.robjects.numpy2ri.activate()
    RRuntimeError = rpy2.robjects.rinterface.RRuntimeError
    r = rpy2.robjects.r
    r.library('lars')
    from mvpa2.support.rpy2_addons import Rrx2

# local imports
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples, \
        FailedToPredictError
from mvpa2.base.learner import FailedToTrainError
from mvpa2.measures.base import Sensitivity
from mvpa2.datasets.base import Dataset

from mvpa2.base import warning
if __debug__:
    from mvpa2.base import debug

known_models = ('lasso', 'stepwise', 'lar', 'forward.stagewise')

class LARS(Classifier):
    """Least angle regression (LARS).

    LARS is the model selection algorithm from:

    Bradley Efron, Trevor Hastie, Iain Johnstone and Robert
    Tibshirani, Least Angle Regression Annals of Statistics (with
    discussion) (2004) 32(2), 407-499. A new method for variable
    subset selection, with the lasso and 'epsilon' forward stagewise
    methods as special cases.

    Similar to SMLR, it performs a feature selection while performing
    classification, but instead of starting with all features, it
    starts with none and adds them in, which is similar to boosting.

    This learner behaves more like a ridge regression in that it
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

    # XXX from yoh: it is linear, isn't it?
    __tags__ = [ 'lars', 'regression', 'linear', 'has_sensitivity',
                 'does_feature_selection', 'rpy2' ]

    def __init__(self, model_type="lasso", trace=False, normalize=True,
                 intercept=True, max_steps=None, use_Gram=False, **kwargs):
        """
        Initialize LARS.

        See the help in R for further details on the following parameters:

        Parameters
        ----------
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
        self.__lowest_Cp_step = None
        self.__weights = None
        """The beta weights for each feature."""
        self.__trained_model = None
        """The model object after training that will be used for
        predictions."""


    def __repr__(self):
        """String summary of the object
        """
        return "LARS(type='%s', normalize=%s, intercept=%s, trace=%s, " \
               "max_steps=%s, use_Gram=%s, " \
               "enable_ca=%s)" % \
               (self.__type,
                self.__normalize,
                self.__intercept,
                self.__trace,
                self.__max_steps,
                self.__use_Gram,
                str(self.ca.enabled))

    @due.dcite(
        Doi('10.1214/009053604000000067'),
        path="mvpa2.clfs.lars:LARS",
        description="Least angle regression",
        tags=["implementation"])
    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """
        targets = data.sa[self.get_space()].value[:, np.newaxis]
        # some non-Python friendly R-lars arguments
        lars_kwargs = {'use.Gram': self.__use_Gram}
        if self.__max_steps is not None:
            lars_kwargs['max.steps'] = self.__max_steps

        trained_model = r.lars(data.samples,
                               targets,
                               type=self.__type,
                               normalize=self.__normalize,
                               intercept=self.__intercept,
                               trace=self.__trace,
                               **lars_kwargs
                               )
        #import pydb
        #pydb.debugger()
        # find the step with the lowest Cp (risk)
        # it is often the last step if you set a max_steps
        # must first convert dictionary to array
        Cp_vals = None
        try:
            Cp_vals = np.asanyarray(Rrx2(trained_model, 'Cp'))
        except TypeError, e:
            raise FailedToTrainError, \
                  "Failed to train %s on %s. Got '%s' while trying to access " \
                  "trained model %s" % (self, data, e, trained_model)

        if Cp_vals is None:
            # if there were no any -- just choose 0th
            lowest_Cp_step = 0
        elif np.isnan(Cp_vals[0]):
            # sometimes may come back nan, so just pick the last one
            lowest_Cp_step = len(Cp_vals)-1
        else:
            # determine the lowest
            lowest_Cp_step = Cp_vals.argmin()

        self.__lowest_Cp_step = lowest_Cp_step
        # set the weights to the lowest Cp step
        self.__weights = np.asanyarray(
            Rrx2(trained_model, 'beta'))[lowest_Cp_step]

        self.__trained_model = trained_model # bind to an instance
#         # set the weights to the final state
#         self.__weights = self.__trained_model['beta'][-1,:]


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict with the final state (i.e., the last step)
        # predict with the lowest Cp step
        try:
            res = r.predict(self.__trained_model,
                            data,
                            mode='step',
                            s=self.__lowest_Cp_step)
                            #s=self.__trained_model['beta'].shape[0])
            fit = np.atleast_1d(Rrx2(res, 'fit'))
        except RRuntimeError, e:
            raise FailedToPredictError, \
                  "Failed to predict on %s using %s. Exceptions was: %s" \
                  % (data, self, e)

        self.ca.estimates = fit
        return fit


    def _init_internals(self):
        """Reinitialize all internals
        """
        self.__lowest_Cp_step = None
        self.__weights = None
        """The beta weights for each feature."""
        self.__trained_model = None
        """The model object after training that will be used for
        predictions."""

    def _untrain(self):
        super(LARS, self)._untrain()
        self._init_internals()


    ##REF: Name was automagically refactored
    def _get_feature_ids(self):
        """Return ids of the used features
        """
        return np.where(np.abs(self.__weights)>0)[0]



    ##REF: Name was automagically refactored
    def get_sensitivity_analyzer(self, **kwargs):
        """Returns a sensitivity analyzer for LARS."""
        return LARSWeights(self, **kwargs)

    weights = property(lambda self: self.__weights)



class LARSWeights(Sensitivity):
    """`SensitivityAnalyzer` that reports the weights LARS trained
    on a given `Dataset`.
    """

    _LEGAL_CLFS = [ LARS ]

    def _call(self, dataset=None):
        """Extract weights from LARS classifier.

        LARS always has weights available, so nothing has to be computed here.
        """
        clf = self.clf
        weights = clf.weights

        if __debug__:
            debug('LARS',
                  "Extracting weights for LARS - "+
                  "Result: min=%f max=%f" %\
                  (np.min(weights), np.max(weights)))

        return Dataset(np.atleast_2d(weights))

