# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Various helpers for making PyMVPA use within IPython more enjoyable
"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals, cfg
if externals.exists('running ipython env'):
    ipython_version = externals.versions['ipython']

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.support.ipython start')

__all__ = ['ipy_activate_pymvpa_goodies']

def _goodies_pre011():
    """Goodies activator for ipython < 0.11
    """
    from IPython.ipapi import get as ipget

    try:
        if not cfg.getboolean('ipython', 'complete protected', False):
            ipget().IP.Completer.omit__names = 2
    finally:
        pass

    if cfg.getboolean('ipython', 'complete collections attributes', True):
        from mvpa2.support.ipython.ipy_pymvpa_completer \
             import activate as ipy_completer_activate
        ipy_completer_activate()

def _goodies_011():
    """Goodies activator for ipython >= 0.11
    """
    from IPython.utils.generics import complete_object

    from mvpa2.base.collections import Collection

    @complete_object.when_type(Collection)
    def complete_collection(obj, prev_completions):
        # Place all the keys into completion for the collection, be it
        # an .a, .sa, .fa, .ca, or .params
        return obj.keys() + prev_completions

def ipy_activate_pymvpa_goodies():
    """Activate PyMVPA additions to IPython

    Currently known goodies (controlled via PyMVPA configuration) are:

    * completions of collections' attributes
    * disabling by default protected attributes of instances in completions
    """
    if ipython_version >= '0.11~':
        _goodies_011()
    else:
        _goodies_pre011()




if __debug__:
    debug('INIT', 'mvpa2.support.ipython end')

