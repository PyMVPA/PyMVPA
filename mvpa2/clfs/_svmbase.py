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

import numpy as np
import textwrap

from mvpa2.support.copy import deepcopy

from mvpa2.base import warning
from mvpa2.base.types import is_sequence_type

from mvpa2.kernels.base import Kernel
from mvpa2.base.dochelpers import handle_docstring, _rst, _rst_section, \
     _rst_indentstr

from mvpa2.clfs.base import Classifier
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureListOf

if __debug__:
    from mvpa2.base import debug


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

    
    _ATTRIBUTE_COLLECTIONS = ['params'] # enforce presence of params collections

    # Placeholder: map kernel names to sensitivity classes, ie
    # 'linear':LinearSVMWeights, for each backend
    _KNOWN_SENSITIVITIES={}
    kernel = Parameter(None,
                       # XXX: Currently, can't be ensured using constraints
                       # allowedtype=Kernel,
                       doc='Kernel object', index=-1)

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
        'tube_epsilon': Parameter(0.01,
                  doc='Epsilon in epsilon-insensitive loss function of '
                      'epsilon-SVM regression (SVR)'),
        'tau': Parameter(1e-6, doc='TAU parameter of KRR regression in shogun'),
        'probability': Parameter(0,
                  doc='Flag to signal either probability estimate is obtained '
                      'within LIBSVM'),
        'shrinking': Parameter(1, doc='Either shrinking is to be conducted'),
        'weight_label': Parameter([], constraints=EnsureListOf(int),
                  doc='To be used in conjunction with weight for custom '
                      'per-label weight'),
        # TODO : merge them into a single dictionary
        'weight': Parameter([], constraints=EnsureListOf(float),
                  doc='Custom weights per label'),
        # For some reason setting up epsilon to 1e-5 slowed things down a bit
        # in comparison to how it was before (in yoh/master) by up to 20%... not clear why
        # may be related to 1e-3 default within _svm.py?
        'epsilon': Parameter(5e-5, min=1e-10,
                  doc='Tolerance of termination criteria. (For nu-SVM default is 0.001)')
        }

    _KNOWN_PARAMS = ()                  # just a placeholder to please lintian
    """Parameters which are specific to a given instantiation of SVM
    """

    __tags__ = [ 'svm', 'kernel-based', 'swig' ]

    def __init__(self, **kwargs):
        """Init base class of SVMs. *Not to be publicly used*

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

        impl, add_params, add_internals, descr = \
              self._KNOWN_IMPLEMENTATIONS[svm_impl]

        # Add corresponding parameters to 'known' depending on the
        # implementation chosen
        if add_params is not None:
            self._KNOWN_PARAMS = \
                 self._KNOWN_PARAMS[:] + list(add_params)


        # Assign per-instance __tags__
        self.__tags__ = self.__tags__[:] + [svm_impl]

        # Add corresponding internals
        if add_internals is not None:
            self.__tags__ += list(add_internals)
        self.__tags__.append(svm_impl)

        k = kwargs.get('kernel', None)
        if k is None:
            kwargs['kernel'] = self.__default_kernel_class__()
        if 'linear' in ('%s'%kwargs['kernel']).lower(): # XXX not necessarily best
            self.__tags__ += [ 'linear', 'has_sensitivity' ]
        else:
            self.__tags__ += [ 'non-linear' ]

        # pop out all args from **kwargs which are known to be SVM parameters
        _args = {}
        for param in self._KNOWN_PARAMS + ['svm_impl']: # Update to remove kp's?
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
                                 (self.__class__, self._KNOWN_PARAMS) + \
                                 list(e.args)[1:]])
            raise e

        # populate collections and add values from arguments
        for paramfamily, paramset in ( (self._KNOWN_PARAMS, self.params),):
            for paramname in paramfamily:
                if not (paramname in self._SVM_PARAMS):
                    raise ValueError, "Unknown parameter %s" % paramname + \
                          ". Known SVM params are: %s" % self._SVM_PARAMS.keys()
                param = deepcopy(self._SVM_PARAMS[paramname])
                if paramname in _args:
                    param.value = _args[paramname]
                    # XXX might want to set default to it -- not just value

                paramset[paramname] = param

        # TODO: Below commented out because kernel_type has been removed.  
        # Find way to set default C as necessary
        
        # tune up C if it has one and non-linear classifier is used
        #if self.params.has_key('C') and kernel_type != "linear" \
               #and self.params['C'].is_default:
            #if __debug__:
                #debug("SVM_", "Assigning default C value to be 1.0 for SVM "
                      #"%s with non-linear kernel" % self)
            #self.params['C'].default = 1.0

        # Some postchecks
        if self.params.has_key('weight') and self.params.has_key('weight_label'):
            if not len(self.params.weight_label) == len(self.params.weight):
                raise ValueError, "Lenghts of 'weight' and 'weight_label' lists " \
                      "must be equal."

            
        if __debug__:
            debug("SVM", "Initialized %s with kernel %s" % 
                  (self, self.params.kernel))


    # XXX RF
    @property
    def kernel_params(self):
        if self.params.kernel:
            return self.params.kernel.params
        return None
    
    def __repr__(self):
        """Definition of the object summary over the object
        """
        res = "%s(svm_impl=%r" % \
              (self.__class__.__name__, self._svm_impl)
        sep = ", "
        # XXX TODO: we should have no kernel_params any longer
        for col in [self.params]:#, self.kernel_params]:
            for k in col.keys():
                # list only params with not default values
                if col[k].is_default: continue
                res += "%s%s=%r" % (sep, k, col[k].value)
                #sep = ', '
        ca = self.ca
        for name, invert in ( ('enable', False), ('disable', True) ):
            ca_chosen = ca._get_enabled(nondefault=False, invert=invert)
            if len(ca_chosen):
                res += sep + "%s_ca=%r" % (name, ca_chosen)

        res += ")"
        return res

    ##REF: Name was automagically refactored
    def _get_cvec(self, data):
        """Estimate default and return scaled by it negative user's C values
        """
        if not self.params.has_key('C'):#svm_type in [_svm.svmc.C_SVC]:
            raise RuntimeError, \
                  "Requested estimation of default C whenever C was not set"

        C = self.params.C
        if not is_sequence_type(C):
            # we were not given a tuple for balancing between classes
            C = [C]

        Cs = list(C[:])               # copy
        for i in xrange(len(Cs)):
            if Cs[i] < 0:
                Cs[i] = self._get_default_c(data.samples)*abs(Cs[i])
                if __debug__:
                    debug("SVM", "Default C for %s was computed to be %s" %
                          (C[i], Cs[i]))

        return Cs

    ##REF: Name was automagically refactored
    def _get_default_c(self, data):
        """Compute default C

        TODO: for non-linear SVMs
        """

        if self.params.kernel.__kernel_name__ == 'linear':
            # TODO: move into a function wrapper for
            #       np.linalg.norm
            if np.issubdtype(data.dtype, np.integer):
                # we are dealing with integers and overflows are
                # possible, so assure working with floats
                def sq_func(x):
                    y = x.astype(float) # copy as float
                    y *= y              # in-place square
                    return y
            else:
                sq_func = np.square
            # perform it per each sample so we do not double memory
            # with calling sq_func on full data
            # Having a list of norms here automagically resolves issue
            # with memmapped operations on which return
            # in turn another memmap
            datasetnorm = np.mean([np.sqrt(np.sum(sq_func(s)))
                                   for s in data])
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


    # TODO: make part of kernel object
    #def _getDefaultGamma(self, dataset):
        #"""Compute default Gamma

        #TODO: unify bloody libsvm interface so it makes use of this function.
        #Now it is computed within SVMModel.__init__
        #"""

        ## TODO: Check validity of this w/ new kernels (ie sg.Rbf has sigma)
        #if self.kernel_params.has_key('gamma'):
            #value = 1.0 / len(dataset.uniquetargets)
            #if __debug__:
                #debug("SVM", "Default Gamma is computed to be %f" % value)
        #else:
            #raise RuntimeError, "Shouldn't ask for default Gamma here"

        #return value

    ##REF: Name was automagically refactored
    def get_sensitivity_analyzer(self, **kwargs):
        """Returns an appropriate SensitivityAnalyzer."""

        sana = self._KNOWN_SENSITIVITIES.get(self.params.kernel.__kernel_name__,
                                             None)
        if sana:
            return sana(self, **kwargs)
        else:
            raise NotImplementedError, \
                  "Sensitivity analyzers for kernel %s is unknown" % \
                  self.params.kernel


    @classmethod
    ##REF: Name was automagically refactored
    def _customize_doc(cls):
        #cdoc_old = cls.__doc__
        # Need to append documentation to __init__ method
        idoc_old = cls.__init__.__doc__

        idoc = """
SVM/SVR definition is dependent on specifying kernel, implementation
type, and parameters for each of them which vary depending on the
choices made.

Desired implementation is specified in ``svm_impl`` argument. Here
is the list if implementations known to this class, along with
specific to them parameters (described below among the rest of
parameters), and what tasks it is capable to deal with
(e.g. regression, binary and/or multiclass classification):

"""
        # XXX Deprecate
        # To not confuse sphinx -- lets avoid Implementations section
        # %s""" % (_rst_section('Implementations'),)


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
            ['\n%s%s : %s' % (_rst_indentstr, k, v[3])
             + NOS(v[1], "\n" + _rst_indentstr + "  Parameters: %s")
             + NOS(v[2], "%s" % _rst(('','\n')[int(len(v[1])>0)], '')
                   + _rst_indentstr + "  Capabilities: %s")
             for k,v in cls._KNOWN_IMPLEMENTATIONS.iteritems()])

        # Describe kernels
        idoc += """

Kernel choice is specified as a kernel instance with kwargument ``kernel``.
Some kernels (e.g. Linear) might allow computation of per feature
sensitivity.

"""
        # XXX Deprecate
        # %s""" % (_rst_section('Kernels'),)

        #idoc += ''.join(
        #    ['\n%s%s' % (_rst_indentstr, k)
        #     + ('', ' : provides sensitivity')[int(v[2] is not None)]
        #     + '\n    ' + NOS(v[1], '%s', 'No parameters')
        #     for k,v in cls._KERNELS.iteritems()])

        # Finally parameters
        NOS.seen += cls._KNOWN_PARAMS# + cls._KNOWN_KERNEL_PARAMS

        idoc += '\n' + _rst_section('Parameters') + '\n' + '\n'.join(
            [v._paramdoc()
             for k,v in cls._SVM_PARAMS.iteritems()
             if k in NOS.seen])

        cls.__dict__['__init__'].__doc__ = handle_docstring(idoc_old) + idoc


# populate names in parameters
for k, v in _SVM._SVM_PARAMS.iteritems():
    v._set_name(k)

