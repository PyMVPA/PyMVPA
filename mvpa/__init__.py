#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""MultiVariate Pattern Analysis"""

if not __debug__:
# TODO: psyco should be moved upstairs anyways
    try:
        import psyco
        psyco.profile()
    except:
        verbose(5, "Psyco online compilation is not enabled in knn")

#from mvpa.dataset import *
