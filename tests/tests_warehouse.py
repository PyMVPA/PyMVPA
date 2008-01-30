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

from mvpa.datasets.dataset import Dataset
from mvpa.clfs.classifier import Classifier
from mvpa.misc.state import Statefull

if __debug__:
    from mvpa.misc import debug

def dumbFeatureDataset():
    data = [[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1],
            [5,0],[5,1],[6,0],[6,1],[7,0],[7,1],[8,0],[8,1],
            [9,0],[9,1],[10,0],[10,1],[11,0],[11,1],[12,0],[12,1]]
    regs = [1 for i in range(8)] \
         + [2 for i in range(8)] \
         + [3 for i in range(8)]

    return Dataset(samples=data, labels=regs)

def normalFeatureDataset(perlabel=50, nlabels=2, nfeatures=4, nchunks=5,
                         means=None, nonbogus_features=None, snr=1.0):
    """Generate a dataset where each label is some normally
    distributed beastie around specified mean (0 if None).

    snr is assuming that signal has std 1.0 so we just divide noise by snr

    Probably it is a generalization of pureMultivariateSignal where
    means=[ [0,1], [1,0] ]

    Specify either means or nonbogus_features so means get assigned
    accordingly
    """

    data = N.random.standard_normal((perlabel*nlabels, nfeatures))/N.sqrt(snr)
    if (means is None) and (not nonbogus_features is None):
        if len(nonbogus_features) > nlabels:
            raise ValueError, "Can't assign simply a feature to a " + \
                  "class: more nonbogus_features than labels"
        means = N.zeros((len(nonbogus_features), nfeatures))
        # pure multivariate -- single bit per feature
        for i in xrange(len(nonbogus_features)):
            means[i, nonbogus_features[i]] = 1.0
    if not means is None:
        # add mean
        data += N.repeat(N.array(means, ndmin=2), perlabel, axis=0)
    labels = N.concatenate([N.repeat(i, perlabel) for i in range(nlabels)])
    chunks = N.concatenate([N.repeat(range(nchunks), perlabel/nchunks) for i in range(nlabels)])
    return Dataset(samples=data, labels=labels, chunks=chunks)

def pureMultivariateSignal(patterns, signal2noise = 1.5, chunks=None):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    %%%%%%%%%
    % O % X %
    %%%%%%%%%
    % X % O %
    %%%%%%%%%
    """

    # start with noise
    data=N.random.normal(size=(4*patterns,2))

    # add signal
    data[:2*patterns,1] += signal2noise

    data[2*patterns:4*patterns,1] -= signal2noise
    data[:patterns,0] -= signal2noise
    data[2*patterns:3*patterns,0] -= signal2noise
    data[patterns:2*patterns,0] += signal2noise
    data[3*patterns:4*patterns,0] += signal2noise

    # two conditions
    regs = [0 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [0 for i in xrange(patterns)]
    regs = N.array(regs)

    return Dataset(samples=data, labels=regs, chunks=chunks)

def getMVPattern(s2n):
    run1 = pureMultivariateSignal(5, s2n, 1)
    run2 = pureMultivariateSignal(5, s2n, 2)
    run3 = pureMultivariateSignal(5, s2n, 3)
    run4 = pureMultivariateSignal(5, s2n, 4)
    run5 = pureMultivariateSignal(5, s2n, 5)
    run6 = pureMultivariateSignal(5, s2n, 6)

    data = run1 + run2 + run3 + run4 + run5 + run6

    return data

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
                    except Exception, e:
                        # Adjust message making it more informative
                        e.__init__("%s on %s = %s" % (str(e), argname, `argvalue`))
                        # Reraise bloody exception ;-)
                        raise
        return do_sweep
    if len(kwargs) > 1:
        raise NotImplementedError
    return unittest_method

