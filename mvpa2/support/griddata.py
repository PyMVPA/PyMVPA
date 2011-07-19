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

import sys
from mvpa2.base import externals

if externals.exists('griddata', raise_=True):
    if __debug__:
        from mvpa2.base import debug

    try:
        if sys.version_info[:2] >= (2, 5):
            # enforce absolute import
            griddata = __import__('griddata', globals(),
                                  locals(), [], 0).griddata
        else:
            # little trick to be able to import 'griddata' package (which
            # has same name)
            oldname = __name__
            # crazy name with close to zero possibility to cause whatever
            __name__ = 'iaugf9zrkjsbdv91'
            try:
                from griddata import griddata
                # restore old settings
                __name__ = oldname
            except ImportError:
                # restore old settings
                __name__ = oldname
                raise
            if __debug__:
                debug('EXT', 'Using python-griddata')
    except ImportError:
        from matplotlib.mlab import griddata
        if __debug__:
            debug('EXT', 'Using matplotlib.mlab.griddata')
