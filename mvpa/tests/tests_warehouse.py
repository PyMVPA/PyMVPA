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

import tempfile
import shutil
import os
import traceback as tbm
import unittest, sys
import numpy as N

from mvpa import cfg, externals
from mvpa.datasets import Dataset
from mvpa.datasets.splitters import OddEvenSplitter
from mvpa.clfs.base import Classifier
from mvpa.misc.state import ClassWithCollections
from mvpa.misc.data_generators import *

__all__ = [ 'datasets', 'sweepargs', 'N', 'unittest', '_all_ca_enabled',
            'saveload_warehouse']

if __debug__:
    from mvpa.base import debug
    __all__.append('debug')

    _all_ca_enabled = 'ENFORCE_STATES_ENABLED' in debug.active
else:
    _all_ca_enabled = False



def sweepargs(**kwargs):
    """Decorator function to sweep over a given set of classifiers

    Parameters
    ----------
    clfs : list of `Classifier`
      List of classifiers to run method on

    Often some unittest method can be ran on multiple classifiers.
    So this decorator aims to do that
    """
    def unittest_method(method):
        def do_sweep(*args_, **kwargs_):
            """Perform sweeping over provided keyword arguments
            """
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
                        argvalue.ca.reset()
                    # update kwargs_
                    kwargs_[argname] = argvalue
                    # do actual call
                    try:
                        if __debug__:
                            debug('TEST', 'Running %s on args=%r and kwargs=%r'
                                  % (method.__name__, args_, kwargs_))
                        method(*args_, **kwargs_)
                    except AssertionError, e:
                        estr = str(e)
                        etype, value, tb = sys.exc_info()
                        # literal representation of exception tb, so
                        # we could group them later on
                        eidstr = '  '.join(
                            [l for l in tbm.format_exception(etype, value, tb)
                             if not ('do_sweep' in l
                                     or 'unittest.py' in l
                                     or 'AssertionError' in l
                                     or 'Traceback (most' in l)])

                        # Store exception information for later on groupping
                        if not eidstr in failed_tests:
                            failed_tests[eidstr] = []

                        sargvalue = str(argvalue)
                        if not (__debug__ and 'TEST' in debug.active):
                            # by default lets make it of sane length
                            if len(sargvalue) > 100:
                                sargvalue = sargvalue[:95] + ' ...'
                        failed_tests[eidstr].append(
                            # skip top-most tb in sweep_args
                            (argname, sargvalue, tb.tb_next, estr))

                        if __debug__:
                            msg = "%s on %s=%s" % (estr, argname, argvalue)
                            debug('TEST', 'Failed unittest: %s\n%s'
                                  % (eidstr, msg))
                    untrain_clf(argvalue)
                    # TODO: handle different levels of unittests properly
                    if cfg.getboolean('tests', 'quick', False):
                        # on TESTQUICK just run test for 1st entry in the list,
                        # the rest are omitted
                        # TODO: proper partitioning of unittests
                        break

            if len(failed_tests):
                # Lets now create a single AssertionError exception
                # which would nicely incorporate all failed exceptions
                multiple = len(failed_tests) != 1 # is it unique?
                # if so, we don't need to reinclude traceback since it
                # would be spitted out anyways below
                estr = ""
                cestr = "lead to failures of unittest %s" % method.__name__
                if multiple:
                    estr += "\n Different scenarios %s "\
                            "(specific tracebacks are below):" % cestr
                else:
                    estr += "\n Single scenario %s:" % cestr
                for ek, els in failed_tests.iteritems():
                    estr += '\n'
                    if multiple:
                        estr += ek
                    estr += "  on\n    %s" % ("    ".join(
                            ["%s=%s%s\n" %
                             (ea, eav,
                              # Why didn't I just do regular for loop? ;)
                              ":\n     ".join([xx for xx in [' ', es]
                                               if xx != '']))
                             for ea, eav, etb, es in els]))
                    # take first one... they all should be identical
                    etb = els[0][2]
                raise AssertionError(estr), None, etb

        do_sweep.func_name = method.func_name
        return do_sweep

    if len(kwargs) > 1:
        raise NotImplementedError, \
              "No sweeping over multiple arguments in sweepargs. Meanwhile " \
              "use two @sweepargs decorators for the test."

    return unittest_method

# Define datasets to be used all over. Split-half later on is used to
# split into training/testing
#
snr_scale = cfg.get_as_dtype('tests', 'snr scale', float, default=1.0)

specs = {'large' : { 'perlabel': 99, 'nchunks': 11,
                     'nfeatures': 20, 'snr': 8 * snr_scale},
         'medium' :{ 'perlabel': 24, 'nchunks': 6,
                     'nfeatures': 14, 'snr': 8 * snr_scale},
         'small' : { 'perlabel': 12, 'nchunks': 4,
                     'nfeatures': 6, 'snr' : 14 * snr_scale} }

# Lets permute upon each invocation of test, so we could possibly
# trigger some funny cases
nonbogus_pool = N.random.permutation([0, 1, 3, 5])

datasets = {}

for kind, spec in specs.iteritems():
    # set of univariate datasets
    for nlabels in [ 2, 3, 4 ]:
        basename = 'uni%d%s' % (nlabels, kind)
        nonbogus_features = nonbogus_pool[:nlabels]

        dataset = normal_feature_dataset(
            nlabels=nlabels,
            nonbogus_features=nonbogus_features,
            **spec)

        oes = OddEvenSplitter()
        splits = [(train, test) for (train, test) in oes(dataset)]
        for i, replication in enumerate( ['test', 'train'] ):
            dataset_ = splits[0][i]
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
    mask[0, 0, 0] = 0
    mask[1, 3, 2] = 0
    ds = Dataset.from_wizard(samples=data, targets=labels, chunks=chunks,
                             mask=mask, space='myspace')
    datasets['3d%s' % kind] = ds


# some additional datasets
datasets['dumb2'] = dumb_feature_binary_dataset()
datasets['dumb'] = dumb_feature_dataset()
# dataset with few invariant features
_dsinv = dumb_feature_dataset()
_dsinv.samples = N.hstack((_dsinv.samples,
                           N.zeros((_dsinv.nsamples, 1)),
                           N.ones((_dsinv.nsamples, 1))))
datasets['dumbinv'] = _dsinv

# Datasets for regressions testing
datasets['sin_modulated'] = multiple_chunks(sin_modulated, 4, 30, 1)
datasets['sin_modulated_test'] = sin_modulated(30, 1, flat=True)

# simple signal for linear regressors
datasets['chirp_linear'] = multiple_chunks(chirp_linear, 6, 50, 10, 2, 0.3, 0.1)
datasets['chirp_linear_test'] = chirp_linear(20, 5, 2, 0.4, 0.1)

datasets['wr1996'] = multiple_chunks(wr1996, 4, 50)
datasets['wr1996_test'] = wr1996(50)


def saveload_warehouse():
    """Store all warehouse datasets into HDF5 and reload them.
    """
    import h5py
    from mvpa.base.hdf5 import obj2hdf, hdf2obj

    tempdir = tempfile.mkdtemp()

    # store the whole datasets warehouse in one hdf5 file
    hdf = h5py.File(os.path.join(tempdir, 'myhdf5.hdf5'), 'w')
    for d in datasets:
        obj2hdf(hdf, datasets[d], d)
    hdf.close()

    hdf = h5py.File(os.path.join(tempdir, 'myhdf5.hdf5'), 'r')
    rc_ds = {}
    for d in hdf:
        rc_ds[d] = hdf2obj(hdf[d])
    hdf.close()

    #cleanup temp dir
    shutil.rmtree(tempdir, ignore_errors=True)

    # return the reconstructed datasets (for use in datasets warehouse)
    return rc_ds


def get_random_rotation(ns, nt=None, data=None):
    """Return some random rotation (or rotation + dim reduction) matrix

    Parameters
    ----------
    ns : int
      Dimensionality of source space
    nt : int, optional
      Dimensionality of target space
    data : array, optional
      Some data (should have rank high enough) to derive
      rotation
    """
    if nt is None:
        nt = ns
    # figure out some "random" rotation
    d = max(ns, nt)
    if data is None:
        data = N.random.normal(size=(d*10, d))
    _u, _s, _vh = N.linalg.svd(data[:, :d])
    R = _vh[:ns, :nt]
    if ns == nt:
        # Test if it is indeed a rotation matrix ;)
        # Lets flip first axis if necessary
        if N.linalg.det(R) < 0:
            R[:, 0] *= -1.0
    return R


if cfg.getboolean('tests', 'use hdf datasets', False):
    if not externals.exists('h5py'):
        raise RuntimeError(
            "Cannot perform HDF5 dump of all datasets in the warehouse, "
            "because 'h5py' is not available")

    datasets = saveload_warehouse()
    print "Replaced all dataset warehouse for HDF5 loaded alternative."
