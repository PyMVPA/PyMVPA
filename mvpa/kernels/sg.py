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

class BasicSGKernel(SGKernel):
    """Class which can handle most shogun kernel types"""
    #_KNOWN_KERNELS={'linear':sgk.LinearKernel, # Figure out shortcuts later
                    #'rbf':sgk.GaussianKernel,
                    #'poly':sgk.PolyKernel,
                    #}
    # Subclasses should add params in the order in which they are called in the
    # shogun constructor.  They should not have any other kernel params
    kernel_impl = Parameter(None, allowedtype=sgk.Kernel,
                            doc="Shogun Kernel class which calculates kernel")
    def _compute(self, d1,d2):
        d1 = SGKernel._data2features(d1)
        d2 = SGKernel._data2features(d2)
        kparams = self.params._getNames()[1:]
        self._k = self.params.kernel_impl(d1, d2, 
                                          *[p[kp].value for p in kparams])

class LinearSGKernel(BasicSGKernel):
    def __init__(self, *args, **kwargs):
        
        BasicSGKernel.__init__(self, *args, kernel_impl=sgk.LinearKernel, 
                               **kwargs)
        
class RbfSGKernel(BasicSGKernel):
    gamma = Parameter(1, doc="Scaling value for gaussian")
    def __init__(self, *args, **kwargs):
        BasicSGKernel.__init__(self, *args, kernel_impl=sgk.GaussianKernel, 
                               **kwargs)
        
class PrecomputedSGKernel(SGKernel):
    # This class can't be handled directly by BasicSGKernel because it never
    # should take data, and never has compute called, etc
    
    # TODO: currently matrix is stored twice, in params.matrix and in _k
    # This was necessary for consistent interface w/ numpy PrecomputedKernel
    # Also, now matrix can be a SGKernel meaning compute will update _k if
    # SGKernel has been updated, since it's a pointer
    matrix = Parameter(None, doc='NP array, SGKernel, or raw shogun kernel')
        
    def compute(self, *args, **kwargs):
        m = self.params.matrix
        if isinstance(m, SGKernel):
            m = m._k
        self._k = sgk.CustomKernel(m)
    