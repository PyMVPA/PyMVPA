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
from mvpa.datasets.maskeddataset import MaskedDataset
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
                    if __debug__:
                        if '_QUICKTEST_' in debug.active:
                            # on TESTQUICK just run test for 1st entry in the list,
                            # the rest are omitted
                            # TODO: proper partitioning of unittests
                            break

        do_sweep.func_name = method.func_name
        return do_sweep

    if len(kwargs) > 1:
        raise NotImplementedError
    return unittest_method

#
# Define datasets to be used all over
#
specs = { 'large' : { 'perlabel' : 180, 'nchunks' : 3 },
          'small' : { 'perlabel' : 2,   'nchunks' : 1 } }
nonbogus_pool = [0, 1, 3, 5]
datasets = {}
nfeatures = 6

for kind, spec in specs.iteritems():
    # set of univariate datasets
    for nlabels in [ 2, 3, 4 ]:
        basename = 'uni%d%s' % (nlabels, kind)
        for replication in ( 'test', 'train' ):
            datasets["%s_%s" % (basename, replication)] = \
                normalFeatureDataset(
                    nlabels=nlabels,
                    nfeatures=nfeatures,
                    nonbogus_features=nonbogus_pool[:nlabels],
                    snr=8, **spec)
        # shortcut
        datasets[basename] = datasets['%s_train' % basename]

    # sample 3D
    total = 2*spec['perlabel']
    nchunks = spec['nchunks']
    data = N.random.standard_normal(( total, 3, 6, 6 ))
    labels = N.concatenate( ( N.repeat( 0, spec['perlabel'] ),
                              N.repeat( 1, spec['perlabel'] ) ) )
    chunks = N.asarray(range(nchunks)*(total/nchunks))
    mask = N.ones( (3, 6, 6) )
    mask[0,0,0] = 0
    mask[1,3,2] = 0
    datasets['3d%s' % kind] = MaskedDataset(samples=data, labels=labels,
                                            chunks=chunks, mask=mask)
