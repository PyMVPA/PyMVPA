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
import mvpa.base.externals as externals


if externals.exists('libsvm'):
    # By default for now we want simply to import all SVMs from libsvm
    from mvpa.clfs import libsvm
    from mvpa.clfs.libsvm.svm import *

if externals.exists('shogun'):
    from mvpa.clfs import sg
    if not 'LinearCSVMC' in locals():
        from mvpa.clfs.sg.svm import *

if not 'LinearCSVMC' in locals():
    raise RuntimeError, "None of SVM implementions libraries was found"

#try:
#    from sg.svm import *
#
#    # Nu-SVMs are not provided by SG thus reverting to libsvm-wrappers
#    from libsvm.svm import LinearNuSVMC, RbfNuSVMC
#
#    # Or just bind them to C-SVMs ;)
#    #LinearNuSVMC = LinearCSVMC
#    #RbfNuSVMC = RbfCSVMC
#except:
#    from mvpa.misc import warning
#    warning("Cannot import shogun libraries. Reverting back to LibSVM wrappers")
#    from libsvm.svm import *
