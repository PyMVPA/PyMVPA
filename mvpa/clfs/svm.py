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

from mvpa.misc import warning

# If we want simply to import all SVMs from libsvm
# from libsvm.svm import *

try:
    from sg.svm import *
    # Nu-SVMs are not provided by SG thus reverting to libsvm-wrappers
    #from libsvm.svm import LinearNuSVMC, RbfNuSVMC
    LinearNuSVMC = LinearCSVMC
except:
    warning("Cannot import shogun libraries. Reverting back to LibSVM wrappers")
    from libsvm.svm import *
