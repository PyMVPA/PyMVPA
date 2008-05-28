#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common to all SVM implementations functionality. For internal use only"""

__docformat__ = 'restructuredtext'

import numpy as N

from copy import deepcopy

from mvpa.misc import warning
from mvpa.clfs.base import Classifier
from mvpa.misc.param import Parameter

if __debug__:
    from mvpa.misc import debug

class _SVM(Classifier):
    """Support Vector Machine Classifier.

    Base class for all external SVM implementations.

    Derived classes should define:

    * _KERNELS: map(dict) should define assignment to a tuple containing
      implementation kernel type, list of parameters adherent to the
      kernel, and sensitivity analyzer e.g.::

      _KERNELS = {
             'linear': (shogun.Kernel.LinearKernel, (), LinearSVMWeights),
             'rbf' :   (shogun.Kernel.GaussianKernel, ('gamma',), None),
             ...
             }

    """

    _ATTRIBUTE_COLLECTIONS = ['params', 'kernel_params'] # enforce presence of params collections

    _SVM_PARAMS = {
        'C' : Parameter(-1.0, descr='Trade-off parameter. High C -- rigid margin SVM'),
        'nu' : Parameter(0.5, min=0.0, max=1.0, descr='Fraction of datapoints within the margin'),
        'cache_size': Parameter(100, descr='Size of the kernel cache, specified in megabytes'),
        'coef0': Parameter(0.5, descr='Offset coefficient in polynomial and sigmoid kernels'),
        'degree': Parameter(3, descr='Degree of polynomial kernel'),
            # init the parameter interface
        'tube_epsilon': Parameter(0.1, descr='Epsilon in epsilon-insensitive loss function of epsilon-SVM regression (SVR)'),
        'gamma': Parameter(0, descr='Scaling (width in RBF) within non-linear kernels'),
        'tau': Parameter(1e-6, descr='TAU parameter of KRR regression in shogun'),
        'max_shift': Parameter(10.0, min=0.0, descr='Maximal shift for SGs GaussianShiftKernel'),
        'shift_step': Parameter(1.0, min=0.0, descr='Shift step for SGs GaussianShiftKernel'),
        'probability': Parameter(0, descr='Flag to signal either probability estimate is obtained within LibSVM'),
        'shrinking': Parameter(1, descr='Either shrinking is to be conducted'),
        'weight_label': Parameter([], descr='???'),
        'weight': Parameter([], descr='???'),
        # For some reason setting up epsilong to 1e-5 slowed things down a bit
        # in comparison to how it was before (in yoh/master) by up to 20%... not clear why
        # may be related to 1e-3 default within _svm.py?
        'epsilon': Parameter(5e-5,
                        min=1e-10,
                        descr='Tolerance of termination criterium')
        }


    _clf_internals = [ 'svm', 'kernel-based' ]

    def __init__(self, kernel_type='linear', **kwargs):
        """Init base class of SVMs. *Not to be publicly used*

        :Parameters:
          kernel_type : basestr
            String must be a valid key for cls._KERNELS

        TODO: handling of parameters might migrate to be generic for
        all classifiers. SVMs are choosen to be testbase for that
        functionality to see how well it would fit.
        """

        kernel_type = kernel_type.lower()
        self._kernel_type_literal = kernel_type
        if not kernel_type in self._KERNELS:
            raise ValueError, "Unknown kernel " + kernel_type

        # Add corresponding kernel parameters to 'known' depending on what
        # kernel was chosen
        if self._KERNELS[kernel_type][1] is not None:
            # XXX need to do only if it is a class variable
            self._KNOWN_KERNEL_PARAMS = \
                 self._KNOWN_KERNEL_PARAMS[:] + list(self._KERNELS[kernel_type][1])

        # Assign per-instance _clf_internals
        self._clf_internals = self._clf_internals[:]
        if kernel_type == 'linear':
            self._clf_internals += [ 'linear', 'has_sensitivity' ]
        else:
            self._clf_internals += [ 'non-linear' ]

        # pop out all args from **kwargs which are known to be SVM parameters
        _args = {}
        for param in self._KNOWN_KERNEL_PARAMS + self._KNOWN_PARAMS:
            if param in kwargs:
                _args[param] = kwargs.pop(param)

        try:
            Classifier.__init__(self, **kwargs)
        except TypeError, e:
            if "__init__() got an unexpected keyword argument " in e.args[0]:
                # TODO: make it even more specific -- if that argument is listed
                # within _SVM_PARAMS
                e.args = tuple( [e.args[0] +
                                 "\n Given SVM instance knows following parameters: %s" %
                                 self._KNOWN_PARAMS +
                                 ", and kernel parameters: %s" %
                                 self._KNOWN_KERNEL_PARAMS] + list(e.args)[1:])
            raise e

        for paramfamily, paramset in ( (self._KNOWN_PARAMS, self.params),
                                       (self._KNOWN_KERNEL_PARAMS, self.kernel_params)):
            for paramname in paramfamily:
                if not (paramname in self._SVM_PARAMS):
                    raise ValueError, "Unknown parameter %s" % paramname + \
                          ". Known SVM params are: %s" % self._SVM_PARAMS.keys()
                param = deepcopy(self._SVM_PARAMS[paramname])
                param.name = paramname
                if paramname in _args:
                    param.value = _args[paramname]
                    # XXX might want to set default to it -- not just value

                paramset.add(param)

        # Some postchecks
        if self.params.isKnown('weight') and self.params.isKnown('weight_label'):
            if not len(self.weight_label) == len(self.weight):
                raise ValueError, "Lenghts of 'weight' and 'weight_label' lists" \
                      "must be equal."

        self._kernel_type = self._KERNELS[kernel_type][0]
        if __debug__:
            debug("SVM", "Initialized %s with kernel %s:%s" % 
                  (id(self), kernel_type, self._kernel_type))


    def __repr__(self):
        """Definition of the object summary over the object
        """
        res = "%s(kernel_type='%s'" % (self.__class__.__name__, self._kernel_type_literal)
        sep = ", "
        for col in [self.params, self.kernel_params]:
            for k in col.names:
                # list only params with not default values
                if col[k].isDefault: continue
                res += "%s%s=%s" % (sep, k, col[k].value)
                #sep = ', '
        for name, invert in ( ('enable', False), ('disable', True) ):
            states = self.states._getEnabled(nondefault=False, invert=invert)
            if len(states):
                res += sep + "%s_states=%s" % (name, str(states))

        res += ")"
        return res


    def _getDefaultC(self, data):
        """Compute default C

        TODO: for non-linear SVMs
        """

        if self._kernel_type_literal == 'linear':
            datasetnorm = N.mean(N.sqrt(N.sum(data*data, axis=1)))
            value = 1.0/(datasetnorm*datasetnorm)
            if __debug__:
                debug("SVM", "Default C computed to be %f" % value)
        else:
            warning("TODO: Computation of default C is not yet implemented" +
                    " for non-linear SVMs. Assigning 1.0")
            value = 1.0

        return value


    def _getDefaultGamma(self, dataset):
        """Compute default Gamma

        TODO: unify bloody libsvm interface so it makes use of this function.
        Now it is computed within SVMModel.__init__
        """

        if self.kernel_params.isKnown('gamma'):
            value = 1.0 / len(dataset.uniquelabels)
            if __debug__:
                debug("SVM", "Default Gamma is computed to be %f" % value)
        else:
            raise RuntimeError, "Shouldn't ask for default Gamma here"

        return value

    def getSensitivityAnalyzer(self, **kwargs):
        """Returns an appropriate SensitivityAnalyzer."""
        sana = self._KERNELS[self._kernel_type_literal][2]
        if sana is not None:
            return sana(self, **kwargs)
        else:
            raise NotImplementedError, \
                  "Sensitivity analyzers for kernel %s is TODO" % \
                  self._kernel_type_literal
