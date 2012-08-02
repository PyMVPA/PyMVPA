# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Importer for the available SVM and SVR machines.

Multiple external libraries implementing Support Vector Machines
(Classification) and Regressions are available: LIBSVM, and shogun.
This module is just a helper to provide default implementation for SVM
depending on the availability of external libraries. By default LIBSVM
implementation is chosen by default, but in any case both libraries
are available through importing from this module::

 > from mvpa2.clfs.svm import sg, libsvm
 > help(sg.SVM)
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


# TODO: handle choices within cfg
_VALID_BACKENDS = ('libsvm', 'shogun', 'sg')
_svm_backend = cfg.get('svm', 'backend', default='libsvm').lower()
if _svm_backend == 'shogun':
    _svm_backend = 'sg'

if not _svm_backend in _VALID_BACKENDS:
    raise ValueError, 'Configuration option svm.backend got invalid value %s.' \
          ' Valid choices are %s' % (_svm_backend, _VALID_BACKENDS)

if __debug__:
    debug('SVM', 'SVM backend is %s' % _svm_backend)

if externals.exists('shogun'):
    from mvpa2.clfs import sg
    SVM = sg.SVM
    # Somewhat cruel hack -- define "SVM" family of kernels as binds
    # to specific default SVM implementation
    # XXX might need RF
    from mvpa2.kernels import sg as ksg
    LinearSVMKernel = ksg.LinearSGKernel
    RbfSVMKernel = ksg.RbfSGKernel

    #if not 'LinearCSVMC' in locals():
    #    from mvpa2.clfs.sg.svm import *

if externals.exists('libsvm'):
    # By default for now we want simply to import all SVMs from libsvm
    from mvpa2.clfs.libsvmc import svm as libsvm
    _NuSVM = libsvm.SVM
    if _svm_backend == 'libsvm' or SVM is None:
        if __debug__ and _svm_backend != 'libsvm' and SVM is None:
            debug('SVM', 'SVM backend %s was not found, so using libsvm'
                  % _svm_backend)
        SVM = libsvm.SVM
        from mvpa2.kernels import libsvm as kls
        LinearSVMKernel = kls.LinearLSKernel
        RbfSVMKernel = kls.RbfLSKernel
    #from mvpa2.clfs.libsvm.svm import *

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

