# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for pylab which would enforce setting of a backend
"""

__docformat__ = 'restructuredtext'

import sys
from mvpa2.base import externals

__all__ = ['pl']

if externals.exists('pylab', raise_=True):
    # Assure that we have correct backend
    externals._set_matplotlib_backend()

    if sys.version_info[:2] >= (2, 5):
        # enforce absolute import
        pl = __import__('pylab', globals(),
                              locals(), [], 0)
    else:
        # little trick to be able to import 'griddata' package (which
        # has same name)
        oldname = __name__
        # crazy name with close to zero possibility to cause whatever
        __name__ = 'iaugf9zrkjsasdf1'
        try:
            import pylab as pl
            # restore old settings
            __name__ = oldname
        except ImportError:
            # restore old settings
            __name__ = oldname
            raise
