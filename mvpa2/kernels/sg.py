# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA shogun-based kernels

Provides interface to kernels defined in shogun toolbox.  Commonly
used kernels are provided with convenience classes: `LinearSGKernel`,
`RbfSGKernel`, `PolySGKernel`.  If you need to use some other shogun
kernel, use `CustomSGKernel` to define one.
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.externals import exists, versions
from mvpa2.kernels.base import Kernel
from mvpa2.base.param import Parameter

if exists('shogun', raise_=True):
    import shogun.Kernel as sgk
    from shogun.Features import RealFeatures
else:
    # Just to please sphinx documentation
    class Bogus(object):
        pass
    sgk = Bogus()
    sgk.LinearKernel = None
    sgk.GaussianKernel = None
    sgk.PolyKernel = None

if __debug__:
    from mvpa2.base import debug

class SGKernel(Kernel):
    """A Kernel object with internal representation in Shogun"""

    def as_sg(self):
        return self

    def as_raw_sg(self):
        return self._k

    def __array__(self):
        return self._k.get_kernel_matrix()

    @staticmethod
    def _data2features(data):
        """Converts data to shogun features"""
        if __debug__:
            debug('KRN_SG',
                  'Converting data of shape %s into shogun RealFeatures'
                  % (data.shape,))
        res = RealFeatures(data.astype(float).T)
        if __debug__:
            debug('KRN_SG', 'Done converting data')

        return res

# Conversion methods
def _as_raw_sg(kernel):
    """Converts directly to a Shogun kernel"""
    return sgk.CustomKernel(kernel.as_raw_np())
def _as_sg(kernel):
    """Converts this kernel to a Shogun-based representation"""
    return PrecomputedSGKernel(matrix=kernel.as_raw_np())
Kernel.add_conversion('sg', _as_sg, _as_raw_sg)


class _BasicSGKernel(SGKernel):
    """Abstract class which can handle most shogun kernel types

    Subclasses can specify new kernels using the following declarations:

      - __kernel_cls__ = Shogun kernel class
      - __kp_order__ = Tuple which specifies the order of kernel params.
        If there is only one kernel param, this is not necessary
    """

    __TODO__ = """
    - Think either normalizer_* should not become proper Parameter.
    """

    def __init__(self, normalizer_cls=None, normalizer_args=None, **kwargs):
        """
        Parameters
        ----------
        normalizer_cls : sg.Kernel.CKernelNormalizer
          Class to use as a normalizer for the kernel.  Will be instantiated
          upon compute().  Only supported for shogun >= 0.6.5.
          By default (if left None) assigns IdentityKernelNormalizer to assure no
          normalization.
        normalizer_args : None or list
          If necessary, provide a list of arguments for the normalizer.
        """
        SGKernel.__init__(self, **kwargs)
        if (normalizer_cls is not None) and (versions['shogun:rev'] < 3377):
            raise ValueError, \
               "Normalizer specification is supported only for sg >= 0.6.5. " \
               "Please upgrade shogun python modular bindings."

        if normalizer_cls is None and exists('sg ge 0.6.5'):
            normalizer_cls = sgk.IdentityKernelNormalizer
        self._normalizer_cls = normalizer_cls

        if normalizer_args is None:
            normalizer_args = []
        self._normalizer_args = normalizer_args

    def _compute(self, d1, d2):
        d1 = SGKernel._data2features(d1)
        d2 = SGKernel._data2features(d2)
        try:
            order = self.__kp_order__
        except AttributeError:
            # XXX may be we could use param.index to have them sorted?
            order = self.params.keys()
        kvals = [self.params[kp].value for kp in order]
        self._k = self.__kernel_cls__(d1, d2, *kvals)

        if self._normalizer_cls:
            self._k.set_normalizer(
                self._normalizer_cls(*self._normalizer_args))


class CustomSGKernel(_BasicSGKernel):
    """Class which can wrap any Shogun kernel and it's kernel parameters
    """
    # TODO: rename args here for convenience?
    def __init__(self, kernel_cls, kernel_params=None, **kwargs):
        """Initialize CustomSGKernel.

        Parameters
        ----------
        kernel_cls : Shogun.Kernel
          Class of a Kernel from Shogun
        kernel_params : list
          Each item in this list should be a tuple of (kernelparamname, value),
          and the order is the explicit order required by the Shogun constructor
        """
        if kernel_params is None:
            kernel_params = []
        self.__kernel_cls__ = kernel_cls # These are normally static

        _BasicSGKernel.__init__(self, **kwargs)
        order = []
        for k, v in kernel_params:
            self.params[k] = Parameter(default=v)
            order.append(k)
        self.__kp_order__ = tuple(order)

class LinearSGKernel(_BasicSGKernel):
    """A basic linear kernel computed via Shogun: K(a,b) = a*b.T"""
    __kernel_cls__ = sgk.LinearKernel
    __kernel_name__ = 'linear'


class RbfSGKernel(_BasicSGKernel):
    """Radial basis function: K(a,b) = exp(-||a-b||**2/sigma)"""
    __kernel_cls__ = sgk.GaussianKernel
    __kernel_name__ = 'rbf'
    sigma = Parameter(1, doc="Width/division parameter for gaussian kernel")

    def __init__(self, **kwargs):
        # Necessary for proper docstring construction
        _BasicSGKernel.__init__(self, **kwargs)


class PolySGKernel(_BasicSGKernel):
    """Polynomial kernel: K(a,b) = (a*b.T + c)**degree
    c is 1 if and only if 'inhomogenous' is True
    """
    __kernel_cls__ = sgk.PolyKernel
    __kernel_name__ = 'poly'
    __kp_order__ = ('degree', 'inhomogenous')
    degree = Parameter(2, constraints='int', doc="Polynomial order of the kernel")
    inhomogenous = Parameter(True, constraints='bool',
                             doc="Whether +1 is added within the expression")

    if not exists('sg ge 0.6.5'):

        use_normalization = Parameter(False, constraints='bool',
                                      doc="Optional normalization")
        __kp_order__ = __kp_order__ + ('use_normalization',)

    def __init__(self, **kwargs):
        # Necessary for proper docstring construction
        _BasicSGKernel.__init__(self, **kwargs)

class PrecomputedSGKernel(SGKernel):
    """A kernel which is precomputed from a numpy array or a Shogun kernel"""
    # This class can't be handled directly by BasicSGKernel because it never
    # should take data, and never has compute called, etc

    # NB: To avoid storing kernel twice, self.params.matrix = self._k once the
    # kernel is 'computed'

    def __init__(self, matrix=None, **kwargs):
        """Initialize PrecomputedSGKernel

        Parameters
        ----------
        matrix : SGKernel or Kernel or ndarray
          Kernel matrix to be used
        """
        # Convert to appropriate kernel for input
        if isinstance(matrix, SGKernel):
            k = matrix._k # Take internal shogun
        elif isinstance(matrix, Kernel):
            k = matrix.as_raw_np() # Convert to NP otherwise
        else:
            # Otherwise SG would segfault ;-)
            k = np.array(matrix)

        SGKernel.__init__(self, **kwargs)

        if versions['shogun:rev'] >= 4455:
            self._k = sgk.CustomKernel(k)
        else:
            raise RuntimeError, \
                  "Cannot create PrecomputedSGKernel using current version" \
                  " of shogun -- please upgrade"
            # Following lines are not effective since we should have
            # also provided data for CK in those earlier versions
            #self._k = sgk.CustomKernel()
            #self._k.set_full_kernel_matrix_from_full(k)

    def compute(self, *args, **kwargs):
        """'Compute' `PrecomputedSGKernel` -- no actual "computation" is done
        """
        pass
