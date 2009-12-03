# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA shogun-based kernels

"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals, warning
from mvpa.kernels.base import Kernel, N
from mvpa.misc.param import Parameter
if externals.exists('shogun', raiseException=True):
    import shogun.Kernel as sgk
    from shogun.Features import RealFeatures

# Monkey patch Kernel for sg conversion required due to import/definition race
def as_sg(kernel):
    """Converts this kernel to a Shogun-based representation"""
    p = PrecomputedSGKernel(matrix=N.array(kernel))
    p.compute()
    return p
Kernel.as_sg = as_sg

class SGKernel(Kernel):
    """A Kernel object with internel representation in Shogun"""

    def as_sg(self):
        return self

    def __array__(self):
        return self._k.get_kernel_matrix()
    
    @staticmethod
    def _data2features(data):
        """Converts data to shogun features"""
        return RealFeatures(data.astype(float).T)

class _BasicSGKernel(SGKernel):
    """Abstract class which can handle most shogun kernel types"""
    #_KNOWN_KERNELS={'linear':sgk.LinearKernel, # Figure out shortcuts later
                    #'rbf':sgk.GaussianKernel,
                    #'poly':sgk.PolyKernel,
                    #}
                    
    # Subclasses can specify new kernels using the following declarations:
    #__kernel_cls__ = Shogun kernel class
    #__kp_order__ = Tuple which specifies the order of kernel params
    # If there is only one kernel param, this is not necessary
    
    def _compute(self, d1, d2):
        d1 = SGKernel._data2features(d1)
        d2 = SGKernel._data2features(d2)
        try:
            order = self.__kp_order__
        except AttributeError:
            order = self.params._getNames()
        kvals = [self.params[kp].value for kp in order]
        self._k = self.__kernel_cls__(d1, d2, *kvals)
        
        # XXX: Not sure if this is always the best thing to do - some kernels
        # by default normalize with specific methods automatically, which
        # may cause issues in CV etc.  eg PolyKernel -- SG
        self._k.set_normalizer(sgk.IdentityKernelNormalizer())

class CustomSGKernel(_BasicSGKernel):
    def __init__(self, kernel_cls, kernelparams=[], **kwargs):
        """Class which can wrap any Shogun kernel and it's kernel parameters
        
        :Parameters:
        kernelparams: list
          Each item in this list should be a tuple of (kernelparamname, value),
          and the order is the explicit order required by the Shogun constructor
        """
        self.__kernel_cls__ = kernel_cls # These are normally static
        
        _BasicSGKernel.__init__(self, **kwargs)
        order = []
        for k,v in kernelparams:
            self.params.add_collectable(Parameter(name=k, default=v))
            order.append(k)
        self.__kp_order__ = tuple(order)
        
class LinearSGKernel(_BasicSGKernel):
    """A basic linear kernel computed via Shogun: K(a,b) = a*b.T"""
    __kernel_cls__ = sgk.LinearKernel
        
class RbfSGKernel(_BasicSGKernel):
    """Radial basis function: K(a,b) = exp(-||a-b||**2/sigma)"""
    __kernel_cls__ = sgk.GaussianKernel
    sigma = Parameter(1, doc="Width/division parameter for gaussian kernel")
        
class PolySGKernel(_BasicSGKernel):
    """Polynomial kernel: K(a,b) = (a*b.T + c)**degree
    c is 1 if and only if 'inhomogenous' is True
    """
    __kernel_cls__ = sgk.PolyKernel
    __kp_order__ = ('degree', 'inhomogenous')
    degree = Parameter(2, allowedtype=int, doc="Polynomial order of the kernel")
    inhomogenous = Parameter(True, allowedtype=bool,
                             doc="Whether +1 is added within the expression")
    
class PrecomputedSGKernel(SGKernel):
    """A kernel which is precomputed from a numpy array or Shogun kernel"""
    # This class can't be handled directly by BasicSGKernel because it never
    # should take data, and never has compute called, etc
    
    # NB: To avoid storing kernel twice, self.params.matrix = self._k once the
    # kernel is 'computed'
    matrix = Parameter(None, doc='NP array, SGKernel, or raw shogun kernel')
        
    def __init__(self, *args, **kwargs):
        SGKernel.__init__(self, *args, **kwargs)
        self.compute() # Makes sure _k is always available
        
    def compute(self, *args, **kwargs):
        if self._k is None:
            m = self.params.matrix
            if isinstance(m, SGKernel):
                m = m._k
            self._k = sgk.CustomKernel(m)
            self.params.matrix = self._k
    