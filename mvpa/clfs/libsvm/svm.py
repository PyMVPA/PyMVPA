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
from mvpa.base import warning
from mvpa.misc.state import StateVariable

from mvpa.clfs.base import Classifier
from mvpa.clfs._svmbase import _SVM
from mvpa.measures.base import Sensitivity

import _svm as svm
from sens import *

if __debug__:
    from mvpa.base import debug

# we better expose those since they are mentioned in docstrings
from svmc import \
     C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, \
     NU_SVR, LINEAR, POLY, RBF, SIGMOID, \
     PRECOMPUTED


class SVM(_SVM):
    """Support Vector Machine Classifier.

    This is a simple interface to the libSVM package.
    """

    # Since this is internal feature of LibSVM, this state variable is present
    # here
    probabilities = StateVariable(enabled=False,
        doc="Estimates of samples probabilities as provided by LibSVM")

    _KERNELS = { "linear":  (svm.svmc.LINEAR, None, LinearSVMWeights),
                 "rbf" :    (svm.svmc.RBF, ('gamma',), None),
                 "poly":    (svm.svmc.POLY, ('gamma', 'degree', 'coef0'), None),
                 "sigmoid": (svm.svmc.SIGMOID, ('gamma', 'coef0'), None),
                 }
    # TODO: Complete the list ;-)

    # TODO p is specific for SVR
    _KNOWN_PARAMS = [ 'epsilon', 'probability', 'shrinking',
                      'weight_label', 'weight']

    _KNOWN_KERNEL_PARAMS = [ 'cache_size' ]

    _KNOWN_IMPLEMENTATIONS = {
        'C_SVC' : (svm.svmc.C_SVC, ('C',),
                   ('binary', 'multiclass'), 'C-SVM classification'),
        'NU_SVC' : (svm.svmc.NU_SVC, ('nu',),
                    ('binary', 'multiclass'), 'nu-SVM classification'),
        'ONE_CLASS' : (svm.svmc.ONE_CLASS, (),
                       ('oneclass',), 'one-class-SVM'),
        'EPSILON_SVR' : (svm.svmc.EPSILON_SVR, ('tube_epsilon',),
                         ('regression',), 'epsilon-SVM regression'),
        'NU_SVR' : (svm.svmc.NU_SVR, ('nu',),
                    ('regression',), 'nu-SVM regression')
        }

    _clf_internals = _SVM._clf_internals + [ 'libsvm' ]

    def __init__(self,
                 kernel_type='linear',
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

        svm_impl = kwargs.get('svm_impl', None)
        # Depending on given arguments, figure out desired SVM
        # implementation
        if svm_impl is None:
            for arg, impl in [ ('tube_epsilon', 'EPSILON_SVR'),
                               ('C', 'C_SVC'),
                               ('nu', 'NU_SVC') ]:
                if kwargs.has_key(arg):
                    svm_impl = impl
                    if __debug__:
                        debug('SVM', 'No implementation was specified. Since '
                              '%s is given among arguments, assume %s' %
                              (arg, impl))
                    break
            if svm_impl is None:
                svm_impl = 'C_SVC'
                if __debug__:
                      debug('SVM', 'Assign C_SVC "by default"')
        kwargs['svm_impl'] = svm_impl

        # init base class
        _SVM.__init__(self, kernel_type, **kwargs)

        self._svm_type = self._KNOWN_IMPLEMENTATIONS[svm_impl][0]

        if 'nu' in self._KNOWN_PARAMS and 'epsilon' in self._KNOWN_PARAMS:
            # overwrite eps param with new default value (information taken from libSVM
            # docs
            self.params['epsilon'].setDefault(0.001)

        self.__model = None
        """Holds the trained SVM."""



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
        TRANSLATEDICT={'epsilon': 'eps',
                       'tube_epsilon': 'p'}
        args = []
        for paramname, param in self.params.items.items() + self.kernel_params.items.items():
            if paramname in TRANSLATEDICT:
                argname = TRANSLATEDICT[paramname]
            elif paramname in svm.SVMParameter.default_parameters:
                argname = paramname
            else:
                if __debug__:
                    debug("SVM_", "Skipping parameter %s since it is not known"
                          "to libsvm" % paramname)
                continue
            args.append( (argname, param.value) )

        # XXX All those parameters should be fetched if present from
        # **kwargs and create appropriate parameters within .params or .kernel_params
        libsvm_param = svm.SVMParameter(
            kernel_type=self._kernel_type,
            svm_type=self._svm_type,
            **dict(args))
        """Store SVM parameters in libSVM compatible format."""

        if self.params.isKnown('C'):#svm_type in [svm.svmc.C_SVC]:
            if self.C < 0:
                newC = self._getDefaultC(dataset.samples)*abs(self.C)
                if __debug__:
                    debug("SVM", "Computed C to be %s for C=%s" % (newC, self.C))
                libsvm_param._setParameter('C', newC)

        self.__model = svm.SVMModel(svmprob, libsvm_param)


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
            if not self.regression and len(self.trained_labels) > 2:
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

            if len(values)>0 and (not self.regression) and len(self.trained_labels) == 2:
                if __debug__:
                    debug("SVM","Forcing values to be ndarray and reshaping " +
                          "them to be 1D vector")
                values = N.asarray(values).reshape(len(values))
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


    def summary(self):
        """Provide quick summary over the SVM classifier"""
        s = super(SVM, self).summary()
        if self.trained:
            s += '\n # of SVs: %d' % self.__model.getTotalNSV()
            try:
                prm = svm.svmc.svm_model_param_get(self.__model.model)
                C = svm.svmc.svm_parameter_C_get(prm)
                # extract information of how many SVs sit inside the margin,
                # i.e. so called 'bounded SVs'
                inside_margin = N.sum(
                    # take 0.99 to avoid rounding issues
                    N.abs(self.__model.getSVCoef()) >= 0.99*svm.svmc.svm_parameter_C_get(prm))
                s += ' #bounded SVs:%d' % inside_margin
                s += ' used C:%5g' % C
            except:
                pass
        return s


    def untrain(self):
        if __debug__:
            debug("SVM", "Untraining %s and destroying libsvm model" % self)
        super(SVM, self).untrain()
        del self.__model
        self.__model = None

    model = property(fget=lambda self: self.__model)
    """Access to the SVM model."""



#class LinearSVM(SVM):
#    """Base class of all linear SVM classifiers that make use of the libSVM
#    package. Still not meant to be used directly.
#    """
#
#    def __init__(self, svm_impl, **kwargs):
#        """The constructor arguments are virtually identical to the ones of
#        the SVM class, except that 'kernel_type' is set to LINEAR.
#        """
#        # init base class
#        SVM.__init__(self, kernel_type='linear',
#                         svm_impl=svm_impl, **kwargs)
#
#
#    def getSensitivityAnalyzer(self, **kwargs):
#        """Returns an appropriate SensitivityAnalyzer."""
#        return LibSVMLinearSVMWeights(self, **kwargs)
#
#

#class LinearNuSVMC(LinearSVM):
#    """Classifier for linear Nu-SVM classification.
#    """
#
#    def __init__(self, **kwargs):
#        """
#        """
#        # init base class
#        LinearSVM.__init__(self, svm_impl='NU_SVC', **kwargs)
#
#
#class LinearCSVMC(LinearSVM):
#    """Classifier for linear C-SVM classification.
#    """
#
#    def __init__(self, **kwargs):
#        """
#        """
#        # init base class
#        LinearSVM.__init__(self, svm_impl='C_SVC', **kwargs)
#
#
#
#class RbfNuSVMC(SVM):
#    """Nu-SVM classifier using a radial basis function kernel.
#    """
#
#    def __init__(self, **kwargs):
#        """
#        """
#        # init base class
#        SVM.__init__(self, kernel_type='rbf',
#                     svm_impl='NU_SVC', **kwargs)
#
#
#class RbfCSVMC(SVM):
#    """C-SVM classifier using a radial basis function kernel.
#    """
#
#    def __init__(self, **kwargs):
#        """
#        """
#        # init base class
#        SVM.__init__(self, kernel_type='rbf',
#                     svm_impl='C_SVC', **kwargs)
#
#
# check if there is a libsvm version with configurable
# noise reduction ;)
if hasattr(svm.svmc, 'svm_set_verbosity'):
    if __debug__ and "LIBSVM" in debug.active:
        debug("LIBSVM", "Setting verbosity for libsvm to 255")
        svm.svmc.svm_set_verbosity(255)
    else:
        svm.svmc.svm_set_verbosity(0)



class LinearSVMWeights(Sensitivity):
    """`SensitivityAnalyzer` for the LIBSVM implementation of a linear SVM.
    """

    biases = StateVariable(enabled=True,
                           doc="Offsets of separating hyperplanes")


    _LEGAL_CLFS = [ SVM ]


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
        rhos = N.asarray(self.clf.model.getRho())

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

        return N.asarray(weights.T)
