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

from mvpa.clfs.classifier import Classifier
from mvpa.misc.state import Statefull

# Define sets of classifiers
from mvpa.clfs.svm import *
from mvpa.clfs.ridge import *
from mvpa.clfs.knn import *

clfs={'LinearSVMC' : [LinearCSVMC(), LinearNuSVMC()],
      'NonLinearSVMC' : [RbfCSVMC(), RbfNuSVMC()],
      'clfs_with_sens' : [LinearCSVMC(), LinearNuSVMC()],
      }

clfs['all'] = clfs['LinearSVMC'] + clfs['NonLinearSVMC'] + [ kNN(k=1), RidgeReg() ]

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
                    if isinstance(argvalue, Statefull):
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
                    except AssertionError, e:
                        # Adjust message making it more informative
                        e.__init__("%s on %s = %s" % (str(e), argname, `argvalue`))
                        # Reraise bloody exception ;-)
                        raise
        return do_sweep
    if len(kwargs) > 1:
        raise NotImplementedError
    return unittest_method

