#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrap the libsvm package into a very simple class interface."""

__docformat__ = 'restructuredtext'

# take care of conditional import of external classifiers
from mvpa.base import warning, cfg, externals
from _svmbase import _SVM

if __debug__:
    from mvpa.base import debug

# default SVM implementation
SVM = None
_NuSVM = None


# TODO: handle choices within cfg
_VALID_BACKENDS = ('libsvm', 'shogun')
default_backend = cfg.get('svm', 'backend', default='libsvm').lower()

if not default_backend in _VALID_BACKENDS:
    raise ValueError, 'Configuration option svm.backend got invalid value %s.' \
          'Valid choices are %s' % (default_backend, _VALID_BACKENDS)

if __debug__:
    debug('SVM', 'Default SVM backend is %s' % default_backend)

if externals.exists('shogun'):
    from mvpa.clfs import sg
    SVM = sg.SVM
    #if not 'LinearCSVMC' in locals():
    #    from mvpa.clfs.sg.svm import *

if externals.exists('libsvm'):
    # By default for now we want simply to import all SVMs from libsvm
    from mvpa.clfs import libsvm
    _NuSVM = libsvm.SVM
    if default_backend == 'libsvm' or SVM is None:
        if __debug__ and default_backend != 'libsvm' and SVM is None:
            debug('SVM',
                  'Default SVM backend %s was not found, so using libsvm'
                  % default_backend)
        SVM = libsvm.SVM
    #from mvpa.clfs.libsvm.svm import *

if SVM is None:
    warning("None of SVM implementions libraries was found")
else:
    _defaultC = _SVM._SVM_PARAMS['C'].default
    _defaultNu = _SVM._SVM_PARAMS['nu'].default

    # Define some convinience classes
    class LinearCSVMC(SVM):
        """C-SVM classifier using linear kernel.

        See help for %s for more details
        """ % SVM.__class__.__name__

        def __init__(self, C=_defaultC, **kwargs):
            """
            """
            # init base class
            SVM.__init__(self, C=C, kernel_type='linear', **kwargs)


    class RbfCSVMC(SVM):
        """C-SVM classifier using a radial basis function kernel.

        See help for %s for more details
        """ % SVM.__class__.__name__

        def __init__(self, C=_defaultC, **kwargs):
            """
            """
            # init base class
            SVM.__init__(self, C=C, kernel_type='RBF', **kwargs)

    if _NuSVM is not None:
        class LinearNuSVMC(_NuSVM):
            """Nu-SVM classifier using linear kernel.

            See help for %s for more details
            """ % _NuSVM.__class__.__name__

            def __init__(self, nu=_defaultNu, **kwargs):
                """
                """
                # init base class
                _NuSVM.__init__(self, nu=nu, kernel_type='linear', **kwargs)

        class RbfNuSVMC(SVM):
            """Nu-SVM classifier using a radial basis function kernel.

            See help for %s for more details
            """ % SVM.__class__.__name__

            def __init__(self, nu=_defaultNu, **kwargs):
                # init base class
                SVM.__init__(self, nu=nu, kernel_type='RBF', **kwargs)

