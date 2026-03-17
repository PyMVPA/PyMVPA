# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Importer for the available SVM and SVR machines.

LIBSVM is used as the SVM implementation library.
This module is just a helper to provide default implementation for SVM
depending on the availability of external libraries::

 > from mvpa2.clfs.svm import libsvm
 > help(libsvm.SVM)

Please refer to particular interface for more documentation about
parameterization and available kernels and implementations.
"""

__docformat__ = 'restructuredtext'

# take care of conditional import of external classifiers
from mvpa2.base import warning, cfg, externals
from mvpa2.clfs._svmbase import _SVM

if __debug__:
    from mvpa2.base import debug

# SVM implementation to be used "by default"
SVM = None
_NuSVM = None


if externals.exists('libsvm'):
    # By default for now we want simply to import all SVMs from libsvm
    from mvpa2.clfs.libsvmc import svm as libsvm
    _NuSVM = libsvm.SVM
    SVM = libsvm.SVM
    from mvpa2.kernels import libsvm as kls
    LinearSVMKernel = kls.LinearLSKernel
    RbfSVMKernel = kls.RbfLSKernel

if SVM is None:
    warning("None of SVM implementation libraries was found")
else:
    _defaultC = _SVM._SVM_PARAMS['C'].default
    _defaultNu = _SVM._SVM_PARAMS['nu'].default

    _edocs = []
    """List containing tuples of classes and docs to be extended"""

    # Define some convenience classes
    class LinearCSVMC(SVM):
        def __init__(self, C=_defaultC, **kwargs):
            SVM.__init__(self, C=C, kernel=LinearSVMKernel(), **kwargs)

    class RbfCSVMC(SVM):
        def __init__(self, C=_defaultC, **kwargs):
            SVM.__init__(self, C=C, kernel=RbfSVMKernel(), **kwargs)

    _edocs += [
        (LinearCSVMC, SVM, "C-SVM classifier using linear kernel."),
        (RbfCSVMC, SVM,
         "C-SVM classifier using a radial basis function kernel")]

    if _NuSVM is not None:
        class LinearNuSVMC(_NuSVM):
            def __init__(self, nu=_defaultNu, **kwargs):
                _NuSVM.__init__(self, nu=nu, kernel=LinearSVMKernel(), **kwargs)

        class RbfNuSVMC(_NuSVM):
            def __init__(self, nu=_defaultNu, **kwargs):
                _NuSVM.__init__(self, nu=nu, kernel=RbfSVMKernel(), **kwargs)

        _edocs += [
            (LinearNuSVMC, _NuSVM, "Nu-SVM classifier using linear kernel."),
            (RbfNuSVMC, _NuSVM,
             "Nu-SVM classifier using a radial basis function kernel")]

    for _c, _pc, _d in _edocs:
        _c.__doc__ = \
            "%s\n\nSee documentation of `%s` for more information" % \
            (_d, _pc.__class__.__name__)

