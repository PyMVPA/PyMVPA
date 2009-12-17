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

import unittest, traceback, sys
import numpy as N

from mvpa import cfg
from mvpa.datasets import Dataset
from mvpa.datasets.splitters import OddEvenSplitter
from mvpa.clfs.base import Classifier
from mvpa.misc.state import ClassWithCollections
from mvpa.misc.data_generators import *

__all__ = [ 'datasets', 'sweepargs', 'N', 'unittest', '_all_states_enabled' ]

if __debug__:
    from mvpa.base import debug
    __all__.append('debug')

    _all_states_enabled = 'ENFORCE_STATES_ENABLED' in debug.active
else:
    _all_states_enabled = False



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
                """Little helper"""
                if isinstance(argvalue, Classifier):
                    # clear classifier after its use -- just to be sure ;-)
                    argvalue.params.retrainable = False
                    argvalue.untrain()

            failed_tests = {}
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
                    except AssertionError, e:
                        estr = str(e)
                        etype, value, tb = sys.exc_info()
                        # literal representation of exception tb, so
                        # we could group them later on
                        eidstr = '  '.join(
                            [l for l in traceback.format_exception(etype, value, tb)
                             if not ('do_sweep' in l or 'unittest.py' in l
                                     or 'AssertionError' in l or 'Traceback (most' in l)])

                        # Store exception information for later on groupping
                        if not eidstr in failed_tests:
                            failed_tests[eidstr] = []

                        failed_tests[eidstr].append(
                            # skip top-most tb in sweep_args
                            (argname, `argvalue`, tb.tb_next, estr))

                        if __debug__:
                            msg = "%s on %s=%s" % (estr, argname, `argvalue`)
                            debug('TEST', 'Failed unittest: %s\n%s' % (eidstr, msg))
                    untrain_clf(argvalue)
                    # TODO: handle different levels of unittests properly
                    if cfg.getboolean('tests', 'quick', False):
                        # on TESTQUICK just run test for 1st entry in the list,
                        # the rest are omitted
                        # TODO: proper partitioning of unittests
                        break

            if len(failed_tests):
                # Lets now create a single AssertionError exception which would nicely
                # incorporate all failed exceptions
                multiple = len(failed_tests) != 1 # is it unique?
                # if so, we don't need to reinclude traceback since it
                # would be spitted out anyways below
                estr = ""
                cestr = "lead to failures of unittest %s" % method.__name__
                if multiple:
                    estr += "\n Different scenarios %s (specific tracebacks are below):" % cestr
                else:
                    estr += "\n Single scenario %s:" % cestr
                for ek, els in failed_tests.iteritems():
                    estr += '\n'
                    if multiple: estr += ek
                    estr += "  on\n    %s" % ("    ".join(
                            ["%s=%s%s\n" % (ea, eav,
                                            # Why didn't I just do regular for loop? ;)
                                            ":\n     ".join([x for x in [' ', es] if x != '']))
                             for ea, eav, etb, es in els]))
                    etb = els[0][2] # take first one... they all should be identical
                raise AssertionError(estr), None, etb

        do_sweep.func_name = method.func_name
        return do_sweep

    if len(kwargs) > 1:
        raise NotImplementedError
    return unittest_method

# Define datasets to be used all over. Split-half later on is used to
# split into training/testing
#
snr_scale = cfg.getAsDType('tests', 'snr scale', float, default=1.0)

specs = {'large' : { 'perlabel': 99, 'nchunks': 11, 'nfeatures': 20, 'snr': 8 * snr_scale},
         'medium' :{ 'perlabel': 24, 'nchunks': 6,  'nfeatures': 14, 'snr': 8 * snr_scale},
         'small' : { 'perlabel': 12, 'nchunks': 4,  'nfeatures': 6, 'snr' : 14 * snr_scale} }
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
    mask = N.ones((3, 6, 6), dtype='bool')
    mask[0,0,0] = 0
    mask[1,3,2] = 0
    ds = Dataset.from_masked(samples=data, labels=labels, chunks=chunks,
                             mask=mask)
    datasets['3d%s' % kind] = ds


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
