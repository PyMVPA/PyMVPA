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

from mvpa.misc import warning
from mvpa.clfs.classifier import Classifier
from mvpa.misc.param import Parameter

if __debug__:
    from mvpa.misc import debug

class _SVM(Classifier):
    """Support Vector Machine Classifier.

    Base class for all external SVM implementations.

    Derived classes should define:

    * KERNELS: map(dict) should define assignment to internal kernel type
      e.g. KERNELS = { 'linear': shogun.Kernel.LinearKernel, ... }

    """

    _ATTRIBUTE_COLLECTIONS = ['params', 'kernel_params'] # enforce presence of params collections

    epsilon = Parameter(5e-5,           # XXX it used to work fine with 1e-5 but after Yariks
                                        # evil RF it got slowed down too much. 
                        min=1e-10,
                        descr='Tolerance of termination criterium')

    def __init__(self, kernel_type='linear', softness=None, **kwargs):
        """Init base class of SVMs. *Not to be publicly used*

        :Parameters:
          kernel_type : basestr
            String must be a valid key for cls.KERNELS
        """
        Classifier.__init__(self, **kwargs)

        kernel_type = kernel_type.lower()
        if kernel_type in self.KERNELS:
            self._kernel_type = self.KERNELS[kernel_type]
            if __debug__:
                debug("SVM", "Initialized %s with kernel %s:%s" % 
                      (id(self), kernel_type, self._kernel_type))
        else:
            raise ValueError, "Unknown kernel %s" % kernel_type

        # XXX might want to create a Parameter with a list of
        # available kernels?
        self._kernel_type_literal = kernel_type

        if softness is not None:
            softness = softness.lower()

        # assign appropriate parameter
        if softness == 'c':
            # Give it C parameter
            self.params.add(Parameter(-1.0,
                                      name='C',
                                      min=1e-10,
                                      descr='Trade-off parameter. High C -- rigid margin SVM'))
        elif softness == 'nu':
            self.params.add(Parameter(0.5,
                                      name='nu',
                                      min=0.0,
                                      max=1.0,
                                      descr='fraction of datapoints within the margin'))


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


