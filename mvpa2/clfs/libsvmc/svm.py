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

import numpy as np

import operator

from mvpa2.base import warning
from mvpa2.base.state import ConditionalAttribute
from mvpa2.base.learner import FailedToTrainError

from mvpa2.clfs.base import accepts_dataset_as_samples, \
     accepts_samples_as_dataset
from mvpa2.clfs._svmbase import _SVM

from mvpa2.clfs.libsvmc import _svm
from mvpa2.kernels.libsvm import LinearLSKernel
from mvpa2.clfs.libsvmc.sens import LinearSVMWeights

if __debug__:
    from mvpa2.base import debug

# we better expose those since they are mentioned in docstrings
# although pylint would not be happy
from mvpa2.clfs.libsvmc._svmc import \
     C_SVC, NU_SVC, EPSILON_SVR, \
     NU_SVR, LINEAR, POLY, RBF, SIGMOID, \
     PRECOMPUTED, ONE_CLASS

def _data2ls(data):
    return np.asarray(data).astype(float)

class SVM(_SVM):
    """Support Vector Machine Classifier.

    This is a simple interface to the libSVM package.
    """

    # Since this is internal feature of LibSVM, this conditional attribute is present
    # here
    probabilities = ConditionalAttribute(enabled=False,
        doc="Estimates of samples probabilities as provided by LibSVM")

    # TODO p is specific for SVR
    _KNOWN_PARAMS = [ 'epsilon', 'probability', 'shrinking',
                      'weight_label', 'weight']

    #_KNOWN_KERNEL_PARAMS = [ 'cache_size' ]

    _KNOWN_SENSITIVITIES = {'linear':LinearSVMWeights,
                            }
    _KNOWN_IMPLEMENTATIONS = {
        'C_SVC' : (_svm.svmc.C_SVC, ('C',),
                   ('binary', 'multiclass', 'oneclass'), 'C-SVM classification'),
        'NU_SVC' : (_svm.svmc.NU_SVC, ('nu',),
                    ('binary', 'multiclass', 'oneclass'), 'nu-SVM classification'),
        'ONE_CLASS' : (_svm.svmc.ONE_CLASS, (),
                       ('oneclass-binary',), 'one-class-SVM'),
        'EPSILON_SVR' : (_svm.svmc.EPSILON_SVR, ('C', 'tube_epsilon'),
                         ('regression',), 'epsilon-SVM regression'),
        'NU_SVR' : (_svm.svmc.NU_SVR, ('nu', 'tube_epsilon'),
                    ('regression', 'oneclass'), 'nu-SVM regression')
        }

    __default_kernel_class__ = LinearLSKernel
    __tags__ = _SVM.__tags__ + [ 'libsvm' ]

    def __init__(self,
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
        _SVM.__init__(self, **kwargs)

        self._svm_type = self._KNOWN_IMPLEMENTATIONS[svm_impl][0]

        if 'nu' in self._KNOWN_PARAMS and 'epsilon' in self._KNOWN_PARAMS:
            # overwrite eps param with new default value (information
            # taken from libSVM docs
            self.params['epsilon']._set_default(0.001)

        self.__model = None
        """Holds the trained SVM."""



    def _train(self, dataset):
        """Train SVM
        """
        targets_sa_name = self.get_space()    # name of targets sa
        targets_sa = dataset.sa[targets_sa_name] # actual targets sa

        # libsvm needs doubles
        src = _data2ls(dataset)

        # libsvm cannot handle literal labels
        labels = self._attrmap.to_numeric(targets_sa.value).tolist()

        svmprob = _svm.SVMProblem(labels, src )

        # Translate few params
        TRANSLATEDICT = {'epsilon': 'eps',
                         'tube_epsilon': 'p'}
        args = []
        for paramname, param in self.params.items() \
                + self.kernel_params.items():
            if paramname in TRANSLATEDICT:
                argname = TRANSLATEDICT[paramname]
            elif paramname in _svm.SVMParameter.default_parameters:
                argname = paramname
            else:
                if __debug__:
                    debug("SVM_", "Skipping parameter %s since it is not known "
                          "to libsvm" % paramname)
                continue
            args.append( (argname, param.value) )

        # ??? All those parameters should be fetched if present from
        # **kwargs and create appropriate parameters within .params or
        # .kernel_params
        libsvm_param = _svm.SVMParameter(
            kernel_type=self.params.kernel.as_raw_ls(),# Just an integer ID
            svm_type=self._svm_type,
            **dict(args))

        """Store SVM parameters in libSVM compatible format."""

        if self.params.has_key('C'):#svm_type in [_svm.svmc.C_SVC]:
            Cs = self._get_cvec(dataset)
            if len(Cs)>1:
                C0 = abs(Cs[0])
                scale = 1.0/(C0)#*np.sqrt(C0))
                # so we got 1 C per label
                uls = self._attrmap.to_numeric(targets_sa.unique)
                if len(Cs) != len(uls):
                    raise ValueError, "SVM was parameterized with %d Cs but " \
                          "there are %d labels in the dataset" % \
                          (len(Cs), len(targets_sa.unique))
                weight = [ c*scale for c in Cs ]
                # All 3 need to be set to take an effect
                libsvm_param._set_parameter('weight', weight)
                libsvm_param._set_parameter('nr_weight', len(weight))
                libsvm_param._set_parameter('weight_label', uls)
            libsvm_param._set_parameter('C', Cs[0])

        try:
            self.__model = _svm.SVMModel(svmprob, libsvm_param)
        except Exception, e:
            raise FailedToTrainError(str(e))


    @accepts_samples_as_dataset
    def _predict(self, data):
        """Predict values for the data
        """
        # libsvm needs doubles
        src = _data2ls(data)
        ca = self.ca

        predictions = [ self.model.predict(p) for p in src ]

        if ca.is_enabled('estimates'):
            if self.__is_regression__:
                estimates = [ self.model.predict_values_raw(p)[0] for p in src ]
            else:
                # if 'trained_targets' are literal they have to be mapped
                if ( np.issubdtype(self.ca.trained_targets.dtype, 'c') or
                     np.issubdtype(self.ca.trained_targets.dtype, 'U') ):
                    trained_targets = self._attrmap.to_numeric(
                            self.ca.trained_targets)
                else:
                    trained_targets = self.ca.trained_targets
                nlabels = len(trained_targets)
                # XXX We do duplicate work. model.predict calls
                # predict_values_raw internally and then does voting or
                # thresholding. So if speed becomes a factor we might
                # want to move out logic from libsvm over here to base
                # predictions on obtined values, or adjust libsvm to
                # spit out values from predict() as well
                if nlabels == 2 and self._svm_impl != 'ONE_CLASS':
                    # Apperently libsvm reorders labels so we need to
                    # track (1,0) values instead of (0,1) thus just
                    # lets take negative reverse
                    estimates = [ self.model.predict_values(p)[(trained_targets[1],
                                                            trained_targets[0])]
                               for p in src ]
                    if len(estimates) > 0:
                        if __debug__:
                            debug("SVM",
                                  "Forcing estimates to be ndarray and reshaping"
                                  " them into 1D vector")
                        estimates = np.asarray(estimates).reshape(len(estimates))
                else:
                    # In multiclass we return dictionary for all pairs
                    # of labels, since libsvm does 1-vs-1 pairs
                    estimates = [ self.model.predict_values(p) for p in src ]
            ca.estimates = estimates

        if ca.is_enabled("probabilities"):
            # XXX Is this really necesssary? yoh don't think so since
            # assignment to ca is doing the same
            #self.probabilities = [ self.model.predict_probability(p)
            #                       for p in src ]
            try:
                ca.probabilities = [ self.model.predict_probability(p)
                                         for p in src ]
            except TypeError:
                warning("Current SVM %s doesn't support probability " %
                        self + " estimation.")
        return predictions


    def summary(self):
        """Provide quick summary over the SVM classifier
        """
        s = super(SVM, self).summary()
        if self.trained:
            s += '\n # of SVs: %d' % self.__model.get_total_n_sv()
            try:
                prm = _svm.svmc.svm_model_param_get(self.__model.model)
                C = _svm.svmc.svm_parameter_C_get(prm)
                # extract information of how many SVs sit inside the margin,
                # i.e. so called 'bounded SVs'
                inside_margin = np.sum(
                    # take 0.99 to avoid rounding issues
                    np.abs(self.__model.get_sv_coef())
                          >= 0.99*_svm.svmc.svm_parameter_C_get(prm))
                s += ' #bounded SVs:%d' % inside_margin
                s += ' used C:%5g' % C
            except:
                pass
        return s


    def _untrain(self):
        """Untrain libsvm's SVM: forget the model
        """
        if __debug__ and "SVM" in debug.active:
            debug("SVM", "Untraining %s and destroying libsvm model" % self)
        super(SVM, self)._untrain()
        del self.__model
        self.__model = None

    model = property(fget=lambda self: self.__model)
    """Access to the SVM model."""


# try to configure libsvm 'noise reduction'. Due to circular imports,
# we can't check externals here since it would not work.
try:
    # if externals.exists('libsvm verbosity control'):
    if __debug__ and "LIBSVM" in debug.active:
        debug("LIBSVM", "Setting verbosity for libsvm to 255")
        _svm.svmc.svm_set_verbosity(255)
    else:
        _svm.svmc.svm_set_verbosity(0)
except AttributeError:
    warning("Available LIBSVM has no way to control verbosity of the output")

# Assign SVM class to limited set of LinearSVMWeights
LinearSVMWeights._LEGAL_CLFS = [SVM]
