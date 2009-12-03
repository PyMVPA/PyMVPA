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

if externals.exists('shogun', raiseException=True):
    import shogun.Kernel as sgk

# Monkey patch Kernel for sg conversion required due to import/definition race
def as_sg(kernel):
    """Converts this kernel to a Shogun-based representation"""
    return SGStaticKernel(N.array(kernel))
Kernel.as_sg = as_sg

class SGKernel(Kernel):
    """A Kernel object with internel representation in Shogun"""
    _KNOWN_KERNELS={'linear':sgk.LinearKernel,
                    'rbf':sgk.GaussianKernel,
                    'poly':sgk.PolyKernel,
                    }
    def __init__(self, sg_cls_or_literal, *args, **kwargs):
        """
        Parameters
        ----------
          sg_cls_or_literal : basestring or shogun kernel class
            The shogun kernel class to instantiate upon compute, or a string
            which matches a known shogun kernel type
        """
        super(SGKernel, self).__init__()
        # TODO: store args/kwargs to initiate sg_cls
        if isinstance(sg_cls_or_literal, str):
            try:
                sg_cls_or_literal = SGKernel._KNOWN_KERNELS[sg_cls_or_literal]
            except KeyError:
                raise ValueError, "Unknown kernel type %"%sg_cls_or_literal
        self._impl = sg_cls_or_literal

    def compute(self, ds1, ds2=None):
        # XXX
        raise NotImplemented, "TODO"

    def as_sg(self):
        return self

    def __array__(self):
        return self._k.get_kernel_matrix()

class SGStaticKernel(SGKernel):
    # This class can't be handled directly by SGKernel because it never
    # should take data, and never has compute called, etc
    def __init__(self, sgkernel_or_nparray):
        self._k = sgk.CustomKernel(sgkernel_or_nparray)
        
    def compute(self, *args, **kwargs):
        pass
    