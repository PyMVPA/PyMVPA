# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Various utilities to help plotting"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

externals.exists("pylab", raiseException=True)
import pylab as P

interactive_backends = ['GTKAgg', 'TkAgg']

# Backends can be modified only prior importing matplotlib, so it is
# safe to just assign current backend right here
mpl_backend = P.matplotlib.get_backend()
mpl_backend_isinteractive = mpl_backend in interactive_backends

if mpl_backend_isinteractive:
    Pioff = P.ioff
    def Pion():
        """Little helper to call P.draw() and P.ion() if backend is interactive
        """
        P.draw()
        P.ion()
else:
    Pioff = Pion = lambda x:None
