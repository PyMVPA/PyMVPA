# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Elastic-Net (ENET) regression classifier."""

__docformat__ = 'restructuredtext'

# system imports
import numpy as N

import mvpa.base.externals as externals

# do conditional to be able to build module reference
if externals.exists('rpy', raiseException=True) and \
   externals.exists('elasticnet', raiseException=True):
    import rpy
    rpy.r.library('elasticnet')


# local imports
from mvpa.clfs.base import Classifier
from mvpa.measures.base import Sensitivity

if __debug__:
    from mvpa.base import debug

class ENET(Classifier):
    """Elastic-Net regression (ENET) `Classifier`.

    Elastic-Net is the model selection algorithm from:

    :ref:`Zou and Hastie (2005) <ZH05>` 'Regularization and Variable
    Selection via the Elastic Net' Journal of the Royal Statistical
    Society, Series B, 67, 301-320.

    Similar to SMLR, it performs a feature selection while performing
    classification, but instead of starting with all features, it
    starts with none and adds them in, which is similar to boosting.

    Unlike LARS it has both L1 and L2 regularization (instead of just
    L1).  This means that while it tries to sparsify the features it
    also tries to keep redundant features, which may be very very good
    for fMRI classification.

    In the true nature of the PyMVPA framework, this algorithm was
    actually implemented in R by Zou and Hastie and wrapped via RPy.
    To make use of ENET, you must have R and RPy installed as well as
    both the lars and elasticnet contributed package. You can install
    the R and RPy with the following command on Debian-based machines:

    sudo aptitude install python-rpy python-rpy-doc r-base-dev

    You can then install the lars and elasticnet package by running R
    as root and calling:

    install.packages()

    """

    _clf_internals = [ 'enet', 'regression', 'linear', 'has_sensitivity',
                       'does_feature_selection'
                       ]
    def __init__(self, lm=1.0, trace=False, normalize=True,
                 intercept=True, max_steps=None, **kwargs):
        """
        Initialize ENET.

        See the help in R for further details on the following parameters:

        :Parameters:
          lm : float
            Penalty parameter.  0 will perform LARS with no ridge regression.
            Default is 1.0.
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
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # set up the params
        self.__lm = lm
        self.__normalize = normalize
        self.__intercept = intercept
        self.__trace = trace
        self.__max_steps = max_steps

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
        return """ENET(lm=%s, normalize=%s, intercept=%s, trace=%s, max_steps=%s, enable_states=%s)""" % \
               (self.__lm,
                self.__normalize,
                self.__intercept,
                self.__trace,
                self.__max_steps,
                str(self.states.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """
        if self.__max_steps is None:
            # train without specifying max_steps
            self.__trained_model = rpy.r.enet(data.samples,
                                              data.labels[:,N.newaxis],
                                              self.__lm,
                                              normalize=self.__normalize,
                                              intercept=self.__intercept,
                                              trace=self.__trace)
        else:
            # train with specifying max_steps
            self.__trained_model = rpy.r.enet(data.samples,
                                              data.labels[:,N.newaxis],
                                              self.__lm,
                                              normalize=self.__normalize,
                                              intercept=self.__intercept,
                                              trace=self.__trace,
                                              max_steps=self.__max_steps)

        # find the step with the lowest Cp (risk)
        # it is often the last step if you set a max_steps
        # must first convert dictionary to array
#         Cp_vals = N.asarray([self.__trained_model['Cp'][str(x)]
#                              for x in range(len(self.__trained_model['Cp']))])
#         self.__lowest_Cp_step = Cp_vals.argmin()
        
        # set the weights to the last step
        self.__weights = N.zeros(data.nfeatures,dtype=self.__trained_model['beta.pure'].dtype)
        ind = N.asarray(self.__trained_model['allset'])-1
        self.__weights[ind] = self.__trained_model['beta.pure'][-1,:]
        
#         # set the weights to the final state
#         self.__weights = self.__trained_model['beta'][-1,:]


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict with the final state (i.e., the last step)
        res = rpy.r.predict_enet(self.__trained_model,
                                 data,
                                 mode='step',
                                 type='fit',
                                 s=self.__trained_model['beta.pure'].shape[0])
                                 #s=self.__lowest_Cp_step)
                                 
        fit = N.asarray(res['fit'])
        if len(fit.shape) == 0:
            # if we just got 1 sample with a scalar
            fit = fit.reshape( (1,) )
        return fit


    def _getFeatureIds(self):
        """Return ids of the used features
        """
        return N.where(N.abs(self.__weights)>0)[0]



    def getSensitivityAnalyzer(self, **kwargs):
        """Returns a sensitivity analyzer for ENET."""
        return ENETWeights(self, **kwargs)

    weights = property(lambda self: self.__weights)



class ENETWeights(Sensitivity):
    """`SensitivityAnalyzer` that reports the weights ENET trained
    on a given `Dataset`.
    """

    _LEGAL_CLFS = [ ENET ]

    def _call(self, dataset=None):
        """Extract weights from ENET classifier.

        ENET always has weights available, so nothing has to be computed here.
        """
        clf = self.clf
        weights = clf.weights

        if __debug__:
            debug('ENET',
                  "Extracting weights for ENET - "+
                  "Result: min=%f max=%f" %\
                  (N.min(weights), N.max(weights)))

        return weights

