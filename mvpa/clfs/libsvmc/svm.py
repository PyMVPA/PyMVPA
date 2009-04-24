# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrap the libsvm package into a very simple class interface."""

__docformat__ = 'restructuredtext'

import numpy as N

import operator

from mvpa.base import warning
from mvpa.misc.state import StateVariable

from mvpa.clfs._svmbase import _SVM

from mvpa.clfs.libsvmc import _svm as svm
from sens import LinearSVMWeights

if __debug__:
    from mvpa.base import debug

# we better expose those since they are mentioned in docstrings
# although pylint would not be happy
from mvpa.clfs.libsvmc._svmc import \
     C_SVC, NU_SVC, EPSILON_SVR, \
     NU_SVR, LINEAR, POLY, RBF, SIGMOID, \
     PRECOMPUTED, ONE_CLASS


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
        'EPSILON_SVR' : (svm.svmc.EPSILON_SVR, ('C', 'tube_epsilon'),
                         ('regression',), 'epsilon-SVM regression'),
        'NU_SVR' : (svm.svmc.NU_SVR, ('nu', 'tube_epsilon'),
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
        """Interface class to LIBSVM classifiers and regressions.

        Default implementation (C/nu/epsilon SVM) is chosen depending
        on the given parameters (C/nu/tube_epsilon).
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
            # overwrite eps param with new default value (information
            # taken from libSVM docs
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
        TRANSLATEDICT = {'epsilon': 'eps',
                         'tube_epsilon': 'p'}
        args = []
        for paramname, param in self.params.items.items() \
                + self.kernel_params.items.items():
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

        # ??? All those parameters should be fetched if present from
        # **kwargs and create appropriate parameters within .params or
        # .kernel_params
        libsvm_param = svm.SVMParameter(
            kernel_type=self._kernel_type,
            svm_type=self._svm_type,
            **dict(args))
        """Store SVM parameters in libSVM compatible format."""

        if self.params.isKnown('C'):#svm_type in [svm.svmc.C_SVC]:
            C = self.params.C
            if not operator.isSequenceType(C):
                # we were not given a tuple for balancing between classes
                C = [C]

            Cs = list(C[:])               # copy
            for i in xrange(len(Cs)):
                if Cs[i] < 0:
                    Cs[i] = self._getDefaultC(dataset.samples)*abs(Cs[i])
                    if __debug__:
                        debug("SVM", "Default C for %s was computed to be %s" %
                              (C[i], Cs[i]))

            libsvm_param._setParameter('C', Cs[0])

            if len(Cs)>1:
                C0 = abs(C[0])
                scale = 1.0/(C0)#*N.sqrt(C0))
                # so we got 1 C per label
                if len(Cs) != len(dataset.uniquelabels):
                    raise ValueError, "SVM was parametrized with %d Cs but " \
                          "there are %d labels in the dataset" % \
                          (len(Cs), len(dataset.uniquelabels))
                weight = [ c*scale for c in Cs ]
                libsvm_param._setParameter('weight', weight)

        self.__model = svm.SVMModel(svmprob, libsvm_param)


    def _predict(self, data):
        """Predict values for the data
        """
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')
        states = self.states

        predictions = [ self.model.predict(p) for p in src ]

        if states.isEnabled("values"):
            if self.regression:
                values = [ self.model.predictValuesRaw(p)[0] for p in src ]
            else:
                trained_labels = self.trained_labels
                nlabels = len(trained_labels)
                # XXX We do duplicate work. model.predict calls
                # predictValuesRaw internally and then does voting or
                # thresholding. So if speed becomes a factor we might
                # want to move out logic from libsvm over here to base
                # predictions on obtined values, or adjust libsvm to
                # spit out values from predict() as well
                if nlabels == 2:
                    # Apperently libsvm reorders labels so we need to
                    # track (1,0) values instead of (0,1) thus just
                    # lets take negative reverse
                    values = [ self.model.predictValues(p)[(trained_labels[1],
                                                            trained_labels[0])]
                               for p in src ]
                    if len(values) > 0:
                        if __debug__:
                            debug("SVM",
                                  "Forcing values to be ndarray and reshaping"
                                  " them into 1D vector")
                        values = N.asarray(values).reshape(len(values))
                else:
                    # In multiclass we return dictionary for all pairs
                    # of labels, since libsvm does 1-vs-1 pairs
                    values = [ self.model.predictValues(p) for p in src ]
            states.values = values

        if states.isEnabled("probabilities"):
            # XXX Is this really necesssary? yoh don't think so since
            # assignment to states is doing the same
            #self.probabilities = [ self.model.predictProbability(p)
            #                       for p in src ]
            try:
                states.probabilities = [ self.model.predictProbability(p)
                                         for p in src ]
            except TypeError:
                warning("Current SVM %s doesn't support probability " %
                        self + " estimation.")
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
                    N.abs(self.__model.getSVCoef())
                          >= 0.99*svm.svmc.svm_parameter_C_get(prm))
                s += ' #bounded SVs:%d' % inside_margin
                s += ' used C:%5g' % C
            except:
                pass
        return s


    def untrain(self):
        """Untrain libsvm's SVM: forget the model
        """
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

# try to configure libsvm 'noise reduction'. Due to circular imports,
# we can't check externals here since it would not work.
try:
    # if externals.exists('libsvm verbosity control'):
    if __debug__ and "LIBSVM" in debug.active:
        debug("LIBSVM", "Setting verbosity for libsvm to 255")
        svm.svmc.svm_set_verbosity(255)
    else:
        svm.svmc.svm_set_verbosity(0)
except AttributeError:
    warning("Available LIBSVM has no way to control verbosity of the output")

# Assign SVM class to limited set of LinearSVMWeights
LinearSVMWeights._LEGAL_CLFS = [SVM]
