# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""GLM-Net (GLMNET) regression and classifier."""

__docformat__ = 'restructuredtext'

# system imports
import numpy as N

import mvpa.base.externals as externals

# do conditional to be able to build module reference
if externals.exists('glmnet', raiseException=True):
    import rpy2.robjects
    import rpy2.robjects.numpy2ri
    RRuntimeError = rpy2.robjects.rinterface.RRuntimeError
    r = rpy2.robjects.r
    r.library('glmnet')

# local imports
from mvpa.base import warning
from mvpa.clfs.base import Classifier, accepts_dataset_as_samples, \
     FailedToTrainError
from mvpa.measures.base import Sensitivity
from mvpa.misc.param import Parameter
from mvpa.datasets.base import Dataset

if __debug__:
    from mvpa.base import debug

def _label2indlist(labels, ulabels):
    """Convert labels to list of unique label indicies starting at 1.
    """

    # allocate for the new one-of-M labels
    new_labels = N.zeros(len(labels), dtype=N.int)

    # loop and convert to one-of-M
    for i, c in enumerate(ulabels):
        new_labels[labels == c] = i+1

    return [str(l) for l in new_labels.tolist()]


def _label2oneofm(labels, ulabels):
    """Convert labels to one-of-M form.

    TODO: Might be useful elsewhere so could migrate into misc/
    """

    # allocate for the new one-of-M labels
    new_labels = N.zeros((len(labels), len(ulabels)))

    # loop and convert to one-of-M
    for i, c in enumerate(ulabels):
        new_labels[labels == c, i] = 1

    return new_labels


class _GLMNET(Classifier):
    """GLM-Net regression (GLMNET) `Classifier`.

    GLM-Net is the model selection algorithm from:

    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization
    Paths for Generalized Linear Models via Coordinate
    Descent. http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf

    To make use of GLMNET, you must have R and RPy2 installed as well
    as the glmnet contributed package. You can install the R and RPy2
    with the following command on Debian-based machines::

      sudo aptitude install python-rpy2 r-base-dev

    You can then install the glmnet package by running R
    as root and calling::

      install.packages()

    """

    __tags__ = [ 'glmnet', 'linear', 'has_sensitivity',
                 'does_feature_selection'
                 ]

    family = Parameter('gaussian',
                       allowedtype='basestring',
                       choices=["gaussian", "multinomial"],
                       doc="""Response type of your labels (either 'gaussian'
                       for regression or 'multinomial' for classification).""")

    alpha = Parameter(1.0, min=0.01, max=1.0, allowedtype='float',
                      doc="""The elastic net mixing parameter.
                      Larger values will give rise to
                      less L2 regularization, with alpha=1.0
                      as a true LASSO penalty.""")

    nlambda = Parameter(100, allowedtype='int', min=1,
                        doc="""Maximum number of lambdas to calculate
                        before stopping if not converged.""")

    standardize = Parameter(True, allowedtype='bool',
                            doc="""Whether to standardize the variables
                            prior to fitting.""")

    thresh = Parameter(1e-4, min=1e-10, max=1.0, allowedtype='float',
             doc="""Convergence threshold for coordinate descent.""")

    pmax = Parameter(None, min=1, allowedtype='None or int',
             doc="""Limit the maximum number of variables ever to be
             nonzero.""")

    maxit = Parameter(100, min=10, allowedtype='int',
             doc="""Maximum number of outer-loop iterations for
             'multinomial' families.""")

    model_type = Parameter('covariance',
                           allowedtype='basestring',
                           choices=["covariance", "naive"],
             doc="""'covariance' saves all inner-products ever
             computed and can be much faster than 'naive'. The
             latter can be more efficient for
             nfeatures>>nsamples situations.""")

    def __init__(self, **kwargs):
        """
        Initialize GLM-Net.

        See the help in R for further details on the parameters
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint friendly initializations
        self.__weights = None
        """The beta weights for each feature."""
        self.__trained_model = None
        """The model object after training that will be used for
        predictions."""

#     def __repr__(self):
#         """String summary of the object
#         """
#         return """ENET(lm=%s, normalize=%s, intercept=%s, trace=%s, max_steps=%s, enable_states=%s)""" % \
#                (self.__lm,
#                 self.__normalize,
#                 self.__intercept,
#                 self.__trace,
#                 self.__max_steps,
#                 str(self.states.enabled))


    def _train(self, dataset):
        """Train the classifier using `data` (`Dataset`).
        """
        # process the labels based on the model family
        if self.params.family == 'gaussian':
            # do nothing, just save the labels as a list
            #labels = dataset.labels.tolist()
            labels = dataset.labels
            pass
        elif self.params.family == 'multinomial':
            # turn lables into list of range values starting at 1
            #labels = _label2indlist(dataset.labels,
            #                        dataset.uniquelabels)
            labels = _label2oneofm(dataset.labels,
                                    dataset.uniquelabels)

        # save some properties of the data/classification
        self._family = self.params.family
        self._ulabels = dataset.uniquelabels.copy()

        # process the pmax
        if self.params.pmax is None:
            # set it to the num features
            pmax = dataset.nfeatures
        else:
            # use the value
            pmax = self.params.pmax

        # train with specifying max_steps
        try:
            self.__trained_model = r.glmnet(dataset.samples,
                                            labels,
                                            family=self.params.family,
                                            alpha=self.params.alpha,
                                            nlambda=self.params.nlambda,
                                            standardize=self.params.standardize,
                                            thresh=self.params.thresh,
                                            pmax=pmax,
                                            maxit=self.params.maxit,
                                            type=self.params.model_type)
        except RRuntimeError, e:
            raise FailedToTrainError, \
                  "Failed to train %s on %s. Got '%s' during call r.glmnet()." \
                  % (self, dataset, e)

        # get the field names of the model
        fnames = N.array(self.__trained_model.getnames())

        # save the lambda of last step
        ind = N.nonzero(fnames=='lambda')[0]
        self.__last_lambda = N.array(self.__trained_model[ind])[-1]

        # set the weights to the last step
        weights = r.coef(self.__trained_model, s=self.__last_lambda)
        if self.params.family == 'multinomial':
            self.__weights = N.hstack([N.array(r['as.matrix'](weights[i]))[1:]
                                       for i in range(len(weights))])
        elif self.params.family == 'gaussian':
            self.__weights = N.array(r['as.matrix'](weights))[1:,0]


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict with standard method
        values = N.array(r.predict(self.__trained_model,
                                   newx=data,
                                   type='link',
                                   s=self.__last_lambda))

        # predict with the final state (i.e., the last step)
        classes = None
        if self.params.family == 'multinomial':
            # remove last dimension of values
            values = values[:, :, 0]

            # get the classes too (they are 1-indexed)
            class_ind = N.array(r.predict(self.__trained_model,
                                          newx=data,
                                          type='class',
                                          s=self.__last_lambda))

            # convert to 0-based ints
            class_ind = (class_ind-1).astype('int')

            # convert to actual labels
            # XXX If just one sample is predicted, the converted predictions
            # array is just 1D, hence it yields an IndexError on [:,0]
            # Modified to .squeeze() which should do the same.
            # Please acknowledge and remove this comment.
            #classes = self._ulabels[class_ind][:,0]
            classes = self._ulabels[class_ind].squeeze()
        else:
            # is gaussian, so just remove last dim of values
            values = values[:, 0]

        # values need to be set anyways if values state is enabled
        self.states.estimates = values
        if classes is not None:
            # set the values and return none
            return classes
        else:
            # return the values as predictions
            return values


    def _getFeatureIds(self):
        """Return ids of the used features
        """
        return N.where(N.abs(self.__weights)>0)[0]


    def getSensitivityAnalyzer(self, **kwargs):
        """Returns a sensitivity analyzer for GLMNET."""
        return GLMNETWeights(self, **kwargs)

    weights = property(lambda self: self.__weights)



class GLMNETWeights(Sensitivity):
    """`SensitivityAnalyzer` that reports the weights GLMNET trained
    on a given `Dataset`.
    """

    _LEGAL_CLFS = [ _GLMNET ]

    def _call(self, dataset=None):
        """Extract weights from GLMNET classifier.

        GLMNET always has weights available, so nothing has to be computed here.
        """
        clf = self.clf
        weights = clf.weights

        if __debug__:
            debug('GLMNET',
                  "Extracting weights for GLMNET - "+
                  "Result: min=%f max=%f" %\
                  (N.min(weights), N.max(weights)))

        #return weights
        if clf._family == 'multinomial':
            return Dataset(weights.T, sa={'labels': clf._ulabels})
        else:
            return weights

class GLMNET_R(_GLMNET):
    """
    GLM-NET Gaussian Regression Classifier.

    This is the GLM-NET algorithm from

    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization
    Paths for Generalized Linear Models via Coordinate
    Descent. http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf

    parameterized to be a regression.

    See GLMNET_C for the multinomial classifier version.

    """

    __tags__ = _GLMNET.__tags__ + ['regression']

    def __init__(self,  **kwargs):
        """
        Initialize GLM-Net.

        See the help in R for further details on the parameters
        """
        # make sure they didn't specify incompatible model
        regr_family = 'gaussian'
        family = kwargs.pop('family', regr_family).lower()
        if family != regr_family:
            warning('You specified the parameter family=%s, but we '
                    'force this to be "%s" for regression.'
                    % (family, regr_family))
            family = regr_family

        # init base class first, forcing regression
        _GLMNET.__init__(self, family=family, **kwargs)


class GLMNET_C(_GLMNET):
    """
    GLM-NET Multinomial Classifier.

    This is the GLM-NET algorithm from

    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization
    Paths for Generalized Linear Models via Coordinate
    Descent. http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf

    parameterized to be a multinomial classifier.

    See GLMNET_Class for the gaussian regression version.

    """

    __tags__ = _GLMNET.__tags__ + ['multiclass', 'binary']

    def __init__(self,  **kwargs):
        """
        Initialize GLM-Net multinomial classifier.

        See the help in R for further details on the parameters
        """
        # make sure they didn't specify regression
        if not kwargs.pop('family', None) is None:
            warning('You specified the "family" parameter, but we '
                    'force this to be "multinomial".')

        # init base class first, forcing regression
        _GLMNET.__init__(self, family='multinomial', **kwargs)

