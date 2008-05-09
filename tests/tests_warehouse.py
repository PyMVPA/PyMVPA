#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscelaneous functions/datasets to be used in the unit tests"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.datasets import Dataset
from mvpa.clfs.base import Classifier
from mvpa.misc.state import Stateful
from mvpa.misc.data_generators import *

if __debug__:
    from mvpa.misc import debug


def sweepargs(**kwargs):
    """Decorator function to sweep over a given set of classifiers

    :Parameters:
      clfs : list of `Classifier`
        List of classifiers to run method on

    Often some unittest method can be ran on multiple classifiers.
    So this decorator aims to do that
    """
    def unittest_method(method):
        def do_sweep(*args_, **kwargs_):
            for argname in kwargs.keys():
                for argvalue in kwargs[argname]:
                    if isinstance(argvalue, Classifier):
                        # clear classifier before its use
                        argvalue.untrain()
                    if isinstance(argvalue, Stateful):
                        argvalue.states.reset()
                    # update kwargs_
                    kwargs_[argname] = argvalue
                    # do actual call
                    try:
                        if __debug__:
                            debug('TEST', 'Running %s on args=%s and kwargs=%s' %
                                  (method.__name__, `args_`, `kwargs_`))
                        method(*args_, **kwargs_)
                        if isinstance(argvalue, Classifier):
                            # clear classifier after its use -- just to be sure ;-)
                            argvalue.untrain()
                    except Exception, e:
                        # Adjust message making it more informative
                        e.__init__("%s on %s = %s" % (str(e), argname, `argvalue`))
                        # Reraise bloody exception ;-)
                        raise
        do_sweep.func_name = method.func_name
        return do_sweep

    if len(kwargs) > 1:
        raise NotImplementedError
    return unittest_method

