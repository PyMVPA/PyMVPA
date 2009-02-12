# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import griddata with preference to the version from matplotlib
"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals
externals.exists('griddata', raiseException=True)

if __debug__:
    from mvpa.base import debug

try:
    from matplotlib.mlab import griddata
except ImportError:
    from griddata import griddata
