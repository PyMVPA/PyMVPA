# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscelaneous functions/datasets to be used in the unit tests"""

__docformat__ = 'restructuredtext'

from os import environ

import unittest
import numpy as N

from mvpa import cfg
from mvpa.datasets import Dataset
from mvpa.datasets.splitters import OddEvenSplitter
from mvpa.datasets.masked import MaskedDataset
from mvpa.clfs.base import Classifier
from mvpa.misc.state import ClassWithCollections
from mvpa.misc.data_generators import *

__all__ = [ 'datasets', 'sweepargs', 'N', 'unittest' ]

if __debug__:
    from mvpa.base import debug
    __all__.append('debug')


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
            def untrain_clf(argvalue):
                if isinstance(argvalue, Classifier):
                    # clear classifier after its use -- just to be sure ;-)
                    argvalue.retrainable = False
                    argvalue.untrain()
            failed_tests_str = []
            exception = None
            for argname in kwargs.keys():
                for argvalue in kwargs[argname]:
                    if isinstance(argvalue, Classifier):
                        # clear classifier before its use
                        argvalue.untrain()
                    if isinstance(argvalue, ClassWithCollections):
                        argvalue.states.reset()
                    # update kwargs_
                    kwargs_[argname] = argvalue
                    # do actual call
                    try:
                        if __debug__:
                            debug('TEST', 'Running %s on args=%s and kwargs=%s' %
                                  (method.__name__, `args_`, `kwargs_`))
                        method(*args_, **kwargs_)
                        untrain_clf(argvalue)
                    except Exception, e:
                        exception = e
                        # Adjust message making it more informative
                        msg = "%s on %s = %s" % (e, argname, `argvalue`)
                        failed_tests_str.append(msg)
                        untrain_clf(argvalue) # untrain classifier
                        if __debug__:
                            debug('TEST', 'Failed #%d: %s' % (len(failed_tests_str), msg))
                    # TODO: handle different levels of unittests properly
                    if cfg.getboolean('tests', 'quick', False):
                        # on TESTQUICK just run test for 1st entry in the list,
                        # the rest are omitted
                        # TODO: proper partitioning of unittests
                        break
            if exception is not None:
                exception.__init__('\n'.join(failed_tests_str))
                raise exception

        do_sweep.func_name = method.func_name
        return do_sweep

    if len(kwargs) > 1:
        raise NotImplementedError
    return unittest_method

# Define datasets to be used all over. Split-half later on is used to
# split into training/testing
#
specs = { 'large' : { 'perlabel' : 99, 'nchunks' : 11, 'nfeatures' : 20, 'snr' : 8 },
          'medium' : { 'perlabel' : 24, 'nchunks' : 6, 'nfeatures' : 14, 'snr' : 8 },
          'small' : { 'perlabel' : 12,  'nchunks' : 4, 'nfeatures' : 6, 'snr' : 14} }
nonbogus_pool = [0, 1, 3, 5]

datasets = {}

for kind, spec in specs.iteritems():
    # set of univariate datasets
    for nlabels in [ 2, 3, 4 ]:
        basename = 'uni%d%s' % (nlabels, kind)
        nonbogus_features=nonbogus_pool[:nlabels]
        bogus_features = filter(lambda x:not x in nonbogus_features,
                                range(spec['nfeatures']))

        dataset = normalFeatureDataset(
            nlabels=nlabels,
            nonbogus_features=nonbogus_features,
            **spec)
        dataset.nonbogus_features = nonbogus_features
        dataset.bogus_features = bogus_features
        oes = OddEvenSplitter()
        splits = [(train, test) for (train, test) in oes(dataset)]
        for i, replication in enumerate( ['test', 'train'] ):
            dataset_ = splits[0][i]
            dataset_.nonbogus_features = nonbogus_features
            dataset_.bogus_features = bogus_features
            datasets["%s_%s" % (basename, replication)] = dataset_

        # full dataset
        datasets[basename] = dataset

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

# some additional datasets
datasets['dumb2'] = dumbFeatureBinaryDataset()
datasets['dumb'] = dumbFeatureDataset()
# dataset with few invariant features
_dsinv = dumbFeatureDataset()
_dsinv.samples = N.hstack((_dsinv.samples,
                           N.zeros((_dsinv.nsamples, 1)),
                           N.ones((_dsinv.nsamples, 1))))
datasets['dumbinv'] = _dsinv

# Datasets for regressions testing
datasets['sin_modulated'] = multipleChunks(sinModulated, 4, 30, 1)
datasets['sin_modulated_test'] = sinModulated(30, 1, flat=True)

# simple signal for linear regressors
datasets['chirp_linear'] = multipleChunks(chirpLinear, 6, 50, 10, 2, 0.3, 0.1)
datasets['chirp_linear_test'] = chirpLinear(20, 5, 2, 0.4, 0.1)

datasets['wr1996'] = multipleChunks(wr1996, 4, 50)
datasets['wr1996_test'] = wr1996(50)
