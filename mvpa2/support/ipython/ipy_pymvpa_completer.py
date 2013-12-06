# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Attributes-aware tab completion.

This module provides a custom tab-completer that intelligently reveals the names
of attributes in the :class:`~mvpa2.base.collections.Collection`\s of PyMVPA.

Activation
==========

To use this, put in your ~/.ipython/ipy_user_conf.py file:

    import ipy_pymvpa_completer as _ic
    _ic.activate()


Usage
=====

The system works as follows.  If t is an `Collection` object, then

In [7]: t.<TAB>

shows not only `dict` interface but also `Attribute`\s present in the collection.

Notes
-----

It is a rip-off from IPython's ipy_traits_completer.py
"""

#############################################################################
# External imports
import mvpa2.base.collections as col
from mvpa2.base import externals

if externals.exists('running ipython env', raise_=True):
    # IPython imports
    from IPython.ipapi import TryNext, get as ipget
    from IPython.genutils import dir2

#############################################################################
# Code begins

def pymvpa_completer(self, event):
    """A custom IPython tab-completer that is collections-aware.
    """

    symbol_parts = event.symbol.split('.')
    base = '.'.join(symbol_parts[:-1])

    oinfo = self._ofind(base)
    if not oinfo['found']:
        raise TryNext

    obj = oinfo['obj']
    # OK, we got the object.  See if it's traits, else punt
    if not isinstance(obj, col.Collection):
        #print "exiting for %s" % obj
        raise TryNext

    # it's a Collection object, lets add its keys
    attrs = dir2(obj)
    #print "adding ", obj.keys()
    attrs += obj.keys()

    # Let's also respect the user's readline_omit__names setting:
    omit__names = ipget().IP.Completer.omit__names
    if omit__names == 1:
        attrs = [a for a in attrs if not a.startswith('__')]
    elif omit__names == 2:
        attrs = [a for a in attrs if not a.startswith('_')]

    # The base of the completion, so we can form the final results list
    bdot = base+'.'

    tcomp = [bdot+a for a in attrs]
    return tcomp

def activate():
    """Activate the PyMVPA Collections completer.
    """
    ipget().set_hook('complete_command', pymvpa_completer, re_key = '.*')


#############################################################################
if __name__ == '__main__':
    # Testing/debugging, can be done only under interactive IPython session
    from mvpa2.datasets.base import dataset_wizard
    t = dataset_wizard([1, 2, 3], targets=1, chunks=2)

    ip = ipget().IP

    assert(not 'targets' in  ip.complete('t.sa.'))
    assert(not 'chunks' in  ip.complete('t.sa.'))

    from ipy_pymvpa_completer import activate
    activate()
    # A few simplistic tests
    assert ip.complete('t.ed') == []
    assert('targets' in  ip.complete('t.sa.'))
    assert('chunks' in  ip.complete('t.sa.'))
    print 'Tests OK'
