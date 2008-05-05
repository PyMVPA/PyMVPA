#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrap the libsvm package into a very simple class interface."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.misc.param import Parameter
from mvpa.misc import warning
from mvpa.misc.state import StateVariable

from mvpa.clfs._svmbase import _SVM
from mvpa.algorithms.datameasure import Sensitivity

import _svm as svm

if __debug__:
    from mvpa.misc import debug

# we better expose those since they are mentioned in docstrings
from svmc import \
     C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, \
     NU_SVR, LINEAR, POLY, RBF, SIGMOID, \
     PRECOMPUTED


class SVMBase(_SVM):
    """Support Vector Machine Classifier.

    This is a simple interface to the libSVM package.
    """

    # Since this is internal feature of LibSVM, this state variable is present
    # here
    probabilities = StateVariable(enabled=False,
        doc="Estimates of samples probabilities as provided by LibSVM")

    _KERNELS = { "linear":  (svm.svmc.LINEAR, None),
                 "rbf" :    (svm.svmc.RBF,     ('gamma',)),
                 "poly":    (svm.svmc.POLY,    ('gamma', 'degree', 'coef0')),
                 "sigmoid": (svm.svmc.SIGMOID, ('gamma', 'coef0')),
                 }
    # TODO: Complete the list ;-)

    # TODO p is specific for SVR
    _KNOWN_PARAMS = [ 'epsilon', 'probability', 'shrinking', 'weight_label', 'weight']

    _KNOWN_KERNEL_PARAMS = [ 'cache_size' ]


    def __init__(self,
                 kernel_type,
                 svm_type,
                 **kwargs):
        # XXX Determine which parameters depend on each other and implement
        # safety/simplifying logic around them
        # already done for: nr_weight
        # thought: weight and weight_label should be a dict
        """This is the base class of all classifier that utilize the libSVM
        package underneath. It is not really meant to be used directly. Unless
        you know what you are doing it is most likely better to use one of the
        subclasses.

        Here is the explaination for some of the parameters from the libSVM
        documentation:

        svm_type can be one of C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR.

        - `C_SVC`: C-SVM classification
        - `NU_SVC`: nu-SVM classification
        - `ONE_CLASS`: one-class-SVM
        - `EPSILON_SVR`: epsilon-SVM regression
        - `NU_SVR`: nu-SVM regression

        kernel_type can be one of LINEAR, POLY, RBF, SIGMOID.

        - `LINEAR`: ``u'*v``
        - `POLY`: ``(gamma*u'*v + coef0)^degree``
        - `RBF`: ``exp(-gamma*|u-v|^2)``
        - `SIGMOID`: ``tanh(gamma*u'*v + coef0)``
        - `PRECOMPUTED`: kernel values in training_set_file

        cache_size is the size of the kernel cache, specified in megabytes.
        C is the cost of constraints violation. (we usually use 1 to 1000)
        eps is the stopping criterion. (we usually use 0.00001 in nu-SVC,
        0.001 in others). nu is the parameter in nu-SVM, nu-SVR, and
        one-class-SVM. p is the epsilon in epsilon-insensitive loss function
        of epsilon-SVM regression. shrinking = 1 means shrinking is conducted;
        = 0 otherwise. probability = 1 means model with probability
        information is obtained; = 0 otherwise.

        nr_weight, weight_label, and weight are used to change the penalty
        for some classes (If the weight for a class is not changed, it is
        set to 1). This is useful for training classifier using unbalanced
        input data or with asymmetric misclassification cost.

        Each weight[i] corresponds to weight_label[i], meaning that
        the penalty of class weight_label[i] is scaled by a factor of weight[i].

        If you do not want to change penalty for any of the classes,
        just set nr_weight to 0.
        """
        self._KNOWN_PARAMS = SVMBase._KNOWN_PARAMS[:]
        self._KNOWN_KERNEL_PARAMS = SVMBase._KNOWN_KERNEL_PARAMS[:]

        if svm_type in [svm.svmc.C_SVC]:
            self._KNOWN_PARAMS += ['C']
        elif svm_type in [svm.svmc.NU_SVC, svm.svmc.NU_SVR]:
            self._KNOWN_PARAMS += ['nu']

        if svm_type in [svm.svmc.EPSILON_SVR]:
            self._KNOWN_PARAMS += ['p']


        # init base class
        _SVM.__init__(self, kernel_type, **kwargs)

        if 'nu' in self._KNOWN_PARAMS:
            # overwrite eps param with new default value (information taken from libSVM
            # docs
            self.params['epsilon'].setDefault(0.001)

        self._svm_type = svm_type

        self.__param = None
        """Holds the trained LibSVM params."""

        self.__model = None
        """Holds the trained SVM."""


    def __repr__(self):
        """Definition of the object summary over the object
        """
        res = "%s(" % self.__class__.__name__
        sep = ""
        for col in [self.params, self.kernel_params]:
            for k in col.names:
                # list only params with not default values
                if col[k].isDefault:
                    continue
                res += "%s%s=%s" % (sep, k, col[k].value)
                sep = ', '

        res += sep + "enable_states=%s" % (str(self.states.enabled))
        res += ")"
        return res


    def _train(self, dataset):
        """Train SVM
        """
        # libsvm needs doubles
        if dataset.samples.dtype == 'float64':
            src = dataset.samples
        else:
            src = dataset.samples.astype('double')

        svmprob = svm.SVMProblem( dataset.labels.tolist(), src )

        # Translate few params
        TRANSLATEDICT={'epsilon': 'eps'}
        args = []
        for paramname, param in self.params.items.items() + self.kernel_params.items.items():
            if paramname in TRANSLATEDICT:
                argname = TRANSLATEDICT[paramname]
            else:
                argname = paramname
            args.append( (argname, param.value) )

        # XXX All those parameters should be fetched if present from
        # **kwargs and create appropriate parameters within .params or .kernel_params
        self.__param = svm.SVMParameter(
                        kernel_type=self._kernel_type,
                        svm_type=self._svm_type,
                        **dict(args))
        """Store SVM parameters in libSVM compatible format."""

        if self.params.isKnown('C'):#svm_type in [svm.svmc.C_SVC]:
            if self.C < 0:
                newC = self._getDefaultC(dataset.samples)*abs(self.C)
                if __debug__:
                    debug("SVM", "Computed C to be %s for C=%s" % (newC, self.C))
                self.__param._setParameter('C', newC)

        self.__model = svm.SVMModel(svmprob, self.__param)


    def _predict(self, data):
        """Predict values for the data
        """
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')

        predictions = [ self.model.predict(p) for p in src ]

        if self.states.isEnabled("values"):
            if len(self.trained_labels) > 2:
                warning("'Values' for multiclass SVM classifier are ambiguous. You " +
                        "are adviced to wrap your classifier with " +
                        "MulticlassClassifier for explicit handling of  " +
                        "separate binary classifiers and corresponding " +
                        "'values'")
            # XXX We do duplicate work. model.predict calls predictValuesRaw
            # internally and then does voting or thresholding. So if speed becomes
            # a factor we might want to move out logic from libsvm over here to base
            # predictions on obtined values, or adjust libsvm to spit out values from
            # predict() as well
            #
            #try:
            values = [ self.model.predictValuesRaw(p) for p in src ]
            if len(values)>0 and len(self.trained_labels) == 2:
                if __debug__:
                    debug("SVM","Forcing values to be ndarray and reshaping " +
                          "them to be 1D vector")
                values = N.array(values).reshape(len(values))
            self.values = values
            # XXX we should probably do the same as shogun for
            # multiclass -- just spit out warning without
            # providing actual values 'per pair' or whatever internal multiclass
            # implementation it was
            #except TypeError:
            #    warning("Current SVM doesn't support probability estimation," +
            #            " thus no 'values' state")

        if self.states.isEnabled("probabilities"):
            self.probabilities = [ self.model.predictProbability(p) for p in src ]
            try:
                self.probabilities = [ self.model.predictProbability(p) for p in src ]
            except TypeError:
                warning("Current SVM %s doesn't support probability estimation," %
                        self + " thus no 'values' state")
        return predictions


    def untrain(self):
        if __debug__:
            debug("SVM", "Untraining %s and destroying libsvm model" % self)
        if self.__param:
            self.__param.untrain()           # reset any automagical assignment of params
        super(SVMBase, self).untrain()
        del self.__model
        self.__model = None

    model = property(fget=lambda self: self.__model)
    """Access to the SVM model."""



class LinearSVM(SVMBase):
    """Base class of all linear SVM classifiers that make use of the libSVM
    package. Still not meant to be used directly.
    """

    def __init__(self, svm_type, **kwargs):
        """The constructor arguments are virtually identical to the ones of
        the SVMBase class, except that 'kernel_type' is set to LINEAR.
        """
        # init base class
        SVMBase.__init__(self, kernel_type='linear',
                         svm_type=svm_type, **kwargs)


    def getSensitivityAnalyzer(self, **kwargs):
        """Returns an appropriate SensitivityAnalyzer."""
        return LibSVMLinearSVMWeights(self, **kwargs)



class LinearNuSVMC(LinearSVM):
    """Classifier for linear Nu-SVM classification.
    """

    def __init__(self, **kwargs):
        """
        """
        # init base class
        LinearSVM.__init__(self, svm_type=svm.svmc.NU_SVC, **kwargs)


class LinearCSVMC(LinearSVM):
    """Classifier for linear C-SVM classification.
    """

    def __init__(self, **kwargs):
        """
        """
        # init base class
        LinearSVM.__init__(self, svm_type=svm.svmc.C_SVC, **kwargs)



class RbfNuSVMC(SVMBase):
    """Nu-SVM classifier using a radial basis function kernel.
    """

    def __init__(self, **kwargs):
        """
        """
        # init base class
        SVMBase.__init__(self, kernel_type='rbf',
                         svm_type=svm.svmc.NU_SVC, **kwargs)


class RbfCSVMC(SVMBase):
    """C-SVM classifier using a radial basis function kernel.
    """

    def __init__(self, **kwargs):
        """
        """
        # init base class
        SVMBase.__init__(self, kernel_type='rbf',
                         svm_type=svm.svmc.C_SVC, **kwargs)


# check if there is a libsvm version with configurable
# noise reduction ;)
if hasattr(svm.svmc, 'svm_set_verbosity'):
    if __debug__ and "SVMLIB" in debug.active:
        debug("SVM", "Setting verbosity for libsvm to 255")
        svm.svmc.svm_set_verbosity(255)
    else:
        svm.svmc.svm_set_verbosity(0)



class LibSVMLinearSVMWeights(Sensitivity):
    """`SensitivityAnalyzer` for the LIBSVM implementation of a linear SVM.
    """

    biases = StateVariable(enabled=True,
                           doc="Offsets of separating hyperplanes")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf: LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        # init base classes first
        Sensitivity.__init__(self, clf, **kwargs)


    def _call(self, dataset, callables=[]):
        if self.clf.model.nr_class != 2:
            warning("You are estimating sensitivity for SVM %s trained on %d" %
                    (str(self.clf), self.clf.model.nr_class) +
                    " classes. Make sure that it is what you intended to do" )

        svcoef = N.matrix(self.clf.model.getSVCoef())
        svs = N.matrix(self.clf.model.getSV())
        rhos = N.array(self.clf.model.getRho())

        self.biases = rhos
        # XXX yoh: .mean() is effectively
        # averages across "sensitivities" of all paired classifiers (I
        # think). See more info on this topic in svm.py on how sv_coefs
        # are stored
        #
        # First multiply SV coefficients with the actuall SVs to get
        # weighted impact of SVs on decision, then for each feature
        # take mean across SVs to get a single weight value
        # per feature
        weights = svcoef * svs

        if __debug__:
            debug('SVM',
                  "Extracting weights for %d-class SVM: #SVs=%s, " % \
                  (self.clf.model.nr_class, str(self.clf.model.getNSV())) + \
                  " SVcoefshape=%s SVs.shape=%s Rhos=%s." % \
                  (svcoef.shape, svs.shape, rhos) + \
                  " Result: min=%f max=%f" % (N.min(weights), N.max(weights)))

        return N.array(weights.T)
