# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for some of NiPy functionality.

Only a handful of functions was borrowed from NiPy until we could rely on
NiPy presence.

Every file borrowed from NiPy copied locally prefixed with _.
support.nipy.__init__ is supposed to serve as a switch board either to import
available NiPy version (if nipy is present and version/revision is good enough)
or a local copy from _'ed file.

"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.support.nipy start')

from mvpa2.base import externals

# NiPy code requires scipy
externals.exists('scipy', raise_=True)
if externals.exists('nipy.neurospin'):
    # Import those interesting ones from nipy
    from nipy.neurospin.utils import emp_null
else:
    import mvpa2.support._emp_null as emp_null

__all__ = ['emp_null']

if __debug__:
    debug('INIT', 'mvpa2.support end')

