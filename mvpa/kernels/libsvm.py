# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA LibSVM-based kernels

These kernels do not currently have the ability to run the calculations, so
they are not translateable to other kernel types.  They are implemented solely
to standardize the interface between other kernel machines.
"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals, warning
from mvpa.kernels.base import Kernel, N
from mvpa.misc.param import Parameter
from mvpa.clfs.libsvmc import _svmc

class LSKernel(Kernel):
    """A Kernel object which dictates how LibSVM will calculate the kernel"""
    
    @property
    def _k(self):
        return self.__kernel_type__
    
    def compute(self, *args, **kwargs):
        pass # Nothing to compute
    
    def as_ls(self):
        return self
    
    def as_np(self):
        raise NotImplemented, 'LibSVM calculates kernels internally; they ' +\
              'cannot be converted to Numpy'


class LinearLSKernel(LSKernel):
    """A simple Linear kernel: K(a,b) = a*b.T"""
    __kernel_type__ = _svmc.LINEAR
    
class RbfLSKernel(LSKernel):
    """Radial Basis Function kernel (aka Gaussian): 
    K(a,b) = exp(-gamma*||a-b||**2)
    """
    __kernel_type__ = _svmc.RBF
    gamma = Parameter(None, doc='Gamma multiplying paramater for Rbf')
    
class PolyLSKernel(LSKernel):
    """Polynomial kernel: K(a,b) = (gamma*a*b.T + coef0)**degree"""
    __kernel_type__ = _svmc.POLY
    gamma = Parameter(None, doc='Gamma multiplying parameter for Polynomial')
    degree = Parameter(None, doc='Degree of polynomial')
    coef0 = Parameter(None, doc='Offset inside polynomial') # aka coef0
    
class SigmoidLSKernel(LSKernel):
    """Sigmoid kernel: K(a,b) = tanh(gamma*a*b.T + coef0)"""
    __kernel_type__ = _svmc.SIGMOID
    gamma = Parameter(None, doc='Gamma multiplying parameter for SigmoidKernel')
    coef0 = Parameter(None, doc='Offset inside tanh')

