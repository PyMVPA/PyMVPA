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

from mvpa2.base import externals

externals.exists("pylab", raise_=True)
import pylab as pl

interactive_backends = ['GTKAgg', 'TkAgg']

# Backends can be modified only prior importing matplotlib, so it is
# safe to just assign current backend right here
mpl_backend = pl.matplotlib.get_backend()
mpl_backend_isinteractive = mpl_backend in interactive_backends

if mpl_backend_isinteractive:
    Pioff = pl.ioff
    def Pion():
        """Little helper to call pl.draw() and pl.ion() if backend is interactive
        """
        pl.draw()
        pl.ion()
else:
    def _Pnothing():
        """Dummy function which does nothing
        """
        pass
    Pioff = Pion = _Pnothing
