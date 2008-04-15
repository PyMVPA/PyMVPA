#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to conditionally import all optional classifier extensions (from
external dependencies)

The global variable 'pymvpa_opt_clf_ext' lists ids of all detected extension
modules. Currently known are:

 * 'libsvm'
 * 'shogun'
"""

__docformat__ = 'restructuredtext'

# contains list of available (optional) external classifier extensions
pymvpa_opt_clf_ext = []

# conditional import of libsvm
try:
    from mvpa.clfs import libsvm
    pymvpa_opt_clf_ext.append('libsvm')
except:
    pass

# conditional import of shogun
try:
    from mvpa.clfs import sg
    pymvpa_opt_clf_ext.append('shogun')
except:
    pass
