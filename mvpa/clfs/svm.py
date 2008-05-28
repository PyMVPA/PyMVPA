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
from mvpa.misc import warning

if externals.exists('libsvm'):
    # By default for now we want simply to import all SVMs from libsvm
    from mvpa.clfs import libsvm
    from mvpa.clfs.libsvm.svm import *

if externals.exists('shogun'):
    from mvpa.clfs import sg
    if not 'LinearCSVMC' in locals():
        from mvpa.clfs.sg.svm import *

if not 'LinearCSVMC' in locals():
    warning("None of SVM implementions libraries was found")
