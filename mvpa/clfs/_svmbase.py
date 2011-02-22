# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common to all SVM implementations functionality. For internal use only"""

__docformat__ = 'restructuredtext'

import numpy as N
import textwrap

from mvpa.support.copy import deepcopy

from mvpa.base import warning
from mvpa.base.dochelpers import handleDocString, _rst, _rst_sep2

from mvpa.clfs.base import Classifier
from mvpa.misc.param import Parameter
from mvpa.misc.transformers import SecondAxisSumOfAbs

if __debug__:
    from mvpa.base import debug


class _SVM(Classifier):
    """Support Vector Machine Classifier.

    Base class for all external SVM implementations.
    """

    """
    Derived classes should define:

    * _KERNELS: map(dict) should define assignment to a tuple containing
      implementation kernel type, list of parameters adherent to the
      kernel, and sensitivity analyzer e.g.::

        _KERNELS = {
             'linear': (shogun.Kernel.LinearKernel, (), LinearSVMWeights),
             'rbf' :   (shogun.Kernel.GaussianKernel, ('gamma',), None),
             ...
             }

    * _KNOWN_IMPLEMENTATIONS: map(dict) should define assignment to a
      tuple containing implementation of the SVM, list of parameters
      adherent to the implementation, additional internals, and
      description e.g.::

        _KNOWN_IMPLEMENTATIONS = {
          'C_SVC' : (svm.svmc.C_SVC, ('C',),
                   ('binary', 'multiclass'), 'C-SVM classification'),
          ...
          }

    """

    _ATTRIBUTE_COLLECTIONS = ['params', 'kernel_params'] # enforce presence of params collections

    _SVM_PARAMS = {
        'C' : Parameter(-1.0,
                  doc='Trade-off parameter between width of the '
                      'margin and number of support vectors. Higher C -- '
                      'more rigid margin SVM. In linear kernel, negative '
                      'values provide automatic scaling of their value '
                      'according to the norm of the data'),
        'nu' : Parameter(0.5, min=0.0, max=1.0,
                  doc='Fraction of datapoints within the margin'),
        'cache_size': Parameter(100,
                  doc='Size of the kernel cache, specified in megabytes'),
        'coef0': Parameter(0.5,
                  doc='Offset coefficient in polynomial and sigmoid kernels'),
        'degree': Parameter(3, doc='Degree of polynomial kernel'),
            # init the parameter interface
        'tube_epsilon': Parameter(0.01,
                  doc='Epsilon in epsilon-insensitive loss function of '
                      'epsilon-SVM regression (SVR)'),
        'gamma': Parameter(0,
                  doc='Scaling (width in RBF) within non-linear kernels'),
        'tau': Parameter(1e-6, doc='TAU parameter of KRR regression in shogun'),
        'max_shift': Parameter(10, min=0.0,
                  doc='Maximal shift for SGs GaussianShiftKernel'),
        'shift_step': Parameter(1, min=0.0,
                  doc='Shift step for SGs GaussianShiftKernel'),
        'probability': Parameter(0,
                  doc='Flag to signal either probability estimate is obtained '
                      'within LIBSVM'),
        'scale': Parameter(1.0,
                  doc='Scale factor for linear kernel. '
                      '(0 triggers automagic rescaling by SG'),
        'shrinking': Parameter(1, doc='Either shrinking is to be conducted'),
        'weight_label': Parameter([], allowedtype='[int]',
                  doc='To be used in conjunction with weight for custom '
                      'per-label weight'),
        # TODO : merge them into a single dictionary
        'weight': Parameter([], allowedtype='[double]',
                  doc='Custom weights per label'),
        # For some reason setting up epsilon to 1e-5 slowed things down a bit
        # in comparison to how it was before (in yoh/master) by up to 20%... not clear why
        # may be related to 1e-3 default within _svm.py?
        'epsilon': Parameter(5e-5, min=1e-10,
                  doc='Tolerance of termination criteria. (For nu-SVM default is 0.001)')
        }


    _clf_internals = [ 'svm', 'kernel-based' ]

    def __init__(self, kernel_type='linear', **kwargs):
        """Init base class of SVMs. *Not to be publicly used*

        :Parameters:
          kernel_type : basestr
            String must be a valid key for cls._KERNELS

        TODO: handling of parameters might migrate to be generic for
        all classifiers. SVMs are chosen to be testbase for that
        functionality to see how well it would fit.
        """

        # Check if requested implementation is known
        svm_impl = kwargs.get('svm_impl', None)
        if not svm_impl in self._KNOWN_IMPLEMENTATIONS:
            raise ValueError, \
                  "Unknown SVM implementation '%s' is requested for %s." \
                  "Known are: %s" % (svm_impl, self.__class__,
                                     self._KNOWN_IMPLEMENTATIONS.keys())
        self._svm_impl = svm_impl

        # Check the kernel
        kernel_type = kernel_type.lower()
        if not kernel_type in self._KERNELS:
            raise ValueError, "Unknown kernel " + kernel_type
        self._kernel_type_literal = kernel_type

        impl, add_params, add_internals, descr = \
              self._KNOWN_IMPLEMENTATIONS[svm_impl]

        # Add corresponding parameters to 'known' depending on the
        # implementation chosen
        if add_params is not None:
            self._KNOWN_PARAMS = \
                 self._KNOWN_PARAMS[:] + list(add_params)

        # Add corresponding kernel parameters to 'known' depending on what
        # kernel chosen
        if self._KERNELS[kernel_type][1] is not None:
            self._KNOWN_KERNEL_PARAMS = \
                 self._KNOWN_KERNEL_PARAMS[:] + list(self._KERNELS[kernel_type][1])

        # Assign per-instance _clf_internals
        self._clf_internals = self._clf_internals[:]

        # Add corresponding internals
        if add_internals is not None:
            self._clf_internals += list(add_internals)
        self._clf_internals.append(svm_impl)

        if kernel_type == 'linear':
            self._clf_internals += [ 'linear', 'has_sensitivity' ]
        else:
            self._clf_internals += [ 'non-linear' ]

        # pop out all args from **kwargs which are known to be SVM parameters
        _args = {}
        for param in self._KNOWN_KERNEL_PARAMS + self._KNOWN_PARAMS + ['svm_impl']:
            if param in kwargs:
                _args[param] = kwargs.pop(param)

        try:
            Classifier.__init__(self, **kwargs)
        except TypeError, e:
            if "__init__() got an unexpected keyword argument " in e.args[0]:
                # TODO: make it even more specific -- if that argument is listed
                # within _SVM_PARAMS
                e.args = tuple( [e.args[0] +
                                 "\n Given SVM instance of class %s knows following parameters: %s" %
                                 (self.__class__, self._KNOWN_PARAMS) +
                                 ", and kernel parameters: %s" %
                                 self._KNOWN_KERNEL_PARAMS] + list(e.args)[1:])
            raise e

        # populate collections and add values from arguments
        for paramfamily, paramset in ( (self._KNOWN_PARAMS, self.params),
                                       (self._KNOWN_KERNEL_PARAMS, self.kernel_params)):
            for paramname in paramfamily:
                if not (paramname in self._SVM_PARAMS):
                    raise ValueError, "Unknown parameter %s" % paramname + \
                          ". Known SVM params are: %s" % self._SVM_PARAMS.keys()
                param = deepcopy(self._SVM_PARAMS[paramname])
                param._setName(paramname)
                if paramname in _args:
                    param.value = _args[paramname]
                    # XXX might want to set default to it -- not just value

                paramset.add(param)

        # tune up C if it has one and non-linear classifier is used
        if self.params.isKnown('C') and kernel_type != "linear" \
               and self.params['C'].isDefault:
            if __debug__:
                debug("SVM_", "Assigning default C value to be 1.0 for SVM "
                      "%s with non-linear kernel" % self)
            self.params['C'].default = 1.0

        # Some postchecks
        if self.params.isKnown('weight') and self.params.isKnown('weight_label'):
            if not len(self.weight_label) == len(self.weight):
                raise ValueError, "Lenghts of 'weight' and 'weight_label' lists " \
                      "must be equal."

        self._kernel_type = self._KERNELS[kernel_type][0]
        if __debug__:
            debug("SVM", "Initialized %s with kernel %s:%s" % 
                  (self, kernel_type, self._kernel_type))


    def __repr__(self):
        """Definition of the object summary over the object
        """
        res = "%s(kernel_type='%s', svm_impl='%s'" % \
              (self.__class__.__name__, self._kernel_type_literal,
               self._svm_impl)
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
            if datasetnorm == 0:
                warning("Obtained degenerate data with zero norm for training "
                        "of %s.  Scaling of C cannot be done." % self)
                return 1.0
            value = 1.0/(datasetnorm**2)
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
            kwargs.setdefault('combiner', SecondAxisSumOfAbs)
            return sana(self, **kwargs)
        else:
            raise NotImplementedError, \
                  "Sensitivity analyzers for kernel %s is TODO" % \
                  self._kernel_type_literal


    @classmethod
    def _customizeDoc(cls):
        #cdoc_old = cls.__doc__
        # Need to append documentation to __init__ method
        idoc_old = cls.__init__.__doc__

        idoc = """
SVM/SVR definition is dependent on specifying kernel, implementation
type, and parameters for each of them which vary depending on the
choices made.

Desired implementation is specified in `svm_impl` argument. Here
is the list if implementations known to this class, along with
specific to them parameters (described below among the rest of
parameters), and what tasks it is capable to deal with
(e.g. regression, binary and/or multiclass classification).

%sImplementations%s""" % (_rst_sep2, _rst_sep2)


        class NOSClass(object):
            """Helper -- NothingOrSomething ;)
            If list is not empty -- return its entries within string s
            """
            def __init__(self):
                self.seen = []
            def __call__(self, l, s, empty=''):
                if l is None or not len(l):
                    return empty
                else:
                    lsorted = list(l)
                    lsorted.sort()
                    self.seen += lsorted
                    return s % (', '.join(lsorted))
        NOS = NOSClass()

        # Describe implementations
        idoc += ''.join(
            ['\n  %s : %s' % (k, v[3])
             + NOS(v[1], "\n    Parameters: %s")
             + NOS(v[2], "\n%s    Capabilities: %%s" %
                   _rst(('','\n')[int(len(v[1])>0)], ''))
             for k,v in cls._KNOWN_IMPLEMENTATIONS.iteritems()])

        # Describe kernels
        idoc += """

Kernel choice is specified as a string argument `kernel_type` and it
can be specialized with additional arguments to this constructor
function. Some kernels might allow computation of per feature
sensitivity.

%sKernels%s""" % (_rst_sep2, _rst_sep2)

        idoc += ''.join(
            ['\n  %s' % k
             + ('', ' : provides sensitivity')[int(v[2] is not None)]
             + '\n    ' + NOS(v[1], '%s', 'No parameters')
             for k,v in cls._KERNELS.iteritems()])

        # Finally parameters
        NOS.seen += cls._KNOWN_PARAMS + cls._KNOWN_KERNEL_PARAMS

        idoc += '\n:Parameters:\n' + '\n'.join(
            [v.doc(indent='  ')
             for k,v in cls._SVM_PARAMS.iteritems()
             if k in NOS.seen])

        cls.__dict__['__init__'].__doc__ = handleDocString(idoc_old) + idoc


# populate names in parameters
for k,v in _SVM._SVM_PARAMS.iteritems():
    v._setName(k)

