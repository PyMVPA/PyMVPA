# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA mode for IPython.
"""

__docformat__ = 'restructuredtext'

from IPython import ipapi

# The import below effectively obsoletes your old-style ipythonrc[.ini],
# so consider yourself warned!
import ipy_defaults

import mvpa2

def main():
    ip = ipapi.get()

    # PyMVPA specific
    ip.ex('import mvpa2')

    # and now the whole suite
    # but no, since ipython segfaults (tested with version 0.8.4)
    # the whole things seems to be related to RPy and friends
    # running the same command after IPython startup is completed
    # is no problem, though.
    #ip.ex('from mvpa2.suite import *')

    print """
###########################
# Welcome to PyMVPA %s #
###########################
""" % mvpa2.__version__

main()
