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
from mvpa.misc.clfhelper import *

if 'libsvm' in pymvpa_opt_clf_ext:
    # By default for now we want simply to import all SVMs from libsvm
    from mvpa.clfs.libsvm.svm import *
elif 'shogun' in pymvpa_opt_clf_ext:
    from mvpa.clfs.sg.svm import *
else:
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
