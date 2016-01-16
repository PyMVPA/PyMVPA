# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helpers for arguments handling."""

__docformat__ = 'restructuredtext'

def split_kwargs(kwargs, prefixes=None):
    """Helper to separate kwargs into multiple groups

    Parameters
    ----------
    prefixes : list of strs
      Each entry sets a prefix which puts entry with key starting
      with it into a separate group.
      Group '' corresponds to 'leftovers'

    :Output:
      dictionary with keys == `prefixes`
    """
    if prefixes is None:
        prefixes = []
    if not ('' in prefixes):
        prefixes = prefixes + ['']
    result = [ [] for i in prefixes ]
    for k,v in kwargs.iteritems():
        for i,p in enumerate(prefixes):
            if k.startswith(p):
                result[i].append((k.replace(p,'',1), v))
                break
    resultd = dict((p,dict(x)) for p,x in zip(prefixes, result))
    return resultd


def group_kwargs(prefixes, assign=False, passthrough=False):
    """Decorator function to join parts of kwargs together

    Parameters
    ----------
    prefixes : list of strs
      Prefixes to split based on. See `split_kwargs`
    assign : bool
      Flag to assign the obtained arguments to self._<prefix>_kwargs
    passthrough : bool
      Flag to pass joined arguments as <prefix>_kwargs argument.
      Usually it is sufficient to have either assign or passthrough.
      If none of those is True, decorator simply filters out mentioned
      groups from being passed to the method

    Example: if needed to join all args which start with 'slave<underscore>'
    together under slave_kwargs parameter
    """
    def decorated_method(method):
        def do_group_kwargs(self, *args_, **kwargs_):
            if '' in prefixes:
                raise ValueError, \
                      "Please don't put empty string ('') into prefixes"
            # group as needed
            splits = split_kwargs(kwargs_, prefixes)
            # adjust resultant kwargs__
            kwargs__ = splits['']
            for prefix in prefixes:
                skwargs = splits[prefix]
                k = '%skwargs' % prefix
                if k in kwargs__:
                    # is unprobable but can happen
                    raise ValueError, '%s is already given in the arguments' % k
                if passthrough:   kwargs__[k] = skwargs
                if assign: setattr(self, '_%s' % k, skwargs)
            return method(self, *args_, **kwargs__)
        do_group_kwargs.func_name = method.func_name
        return do_group_kwargs

    return decorated_method

