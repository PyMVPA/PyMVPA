# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for storage of classifiers in HDF5'''


from mvpa2.testing import *
skip_if_no_external('h5py')
skip_if_no_external('scipy')

import numpy as np
from mvpa2.base import cfg
from mvpa2.testing.datasets import datasets
from mvpa2.clfs.base import Classifier
from mvpa2.generators.splitters import Splitter
from mvpa2.measures.base import TransferMeasure
from mvpa2.misc.errorfx import corr_error, mean_mismatch_error
from mvpa2.mappers.fx import BinaryFxNode

from mvpa2.clfs.warehouse import clfswh, regrswh

import tempfile

from mvpa2.base.hdf5 import h5save, h5load, obj2hdf


@sweepargs(lrn=clfswh[:] + regrswh[:])
def test_h5py_clfs(lrn):
    # lets simply clone it so we could make its all states on
    lrn = lrn.clone()
    # Lets enable all the states
    lrn.ca.enable('all')

    f = tempfile.NamedTemporaryFile()

    # Store/reload untrained learner
    try:
        h5save(f.name, lrn)
    except Exception, e:
        raise AssertionError, \
              "Failed to store due to %r" % (e,)

    try:
        lrn_ = h5load(f.name)
        pass
    except Exception, e:
        raise AssertionError, \
              "Failed to load due to %r" % (e,)

    ok_(isinstance(lrn_, Classifier))
    # Verify that we have the same ca enabled
    # XXX FAILS atm!
    #ok_(set(lrn.ca.enabled) == set(lrn_.ca.enabled))

    # lets choose a dataset
    dsname, errorfx = \
            {False: ('uni2large', mean_mismatch_error),
             True: ('sin_modulated', corr_error)}\
            ['regression' in lrn.__tags__]
    ds = datasets[dsname]
    splitter = Splitter('train')
    postproc = BinaryFxNode(errorfx, 'targets')
    te = TransferMeasure(lrn, splitter, postproc=postproc)
    te_ = TransferMeasure(lrn_, splitter, postproc=postproc)

    error = te(ds)
    error_ = te_(ds)


    if len(set(['swig', 'rpy2']).intersection(lrn.__tags__)):
        raise SkipTest("Trained swigged and R-interfaced classifiers can't "
                       "be stored/reloaded yet")

    # now lets store/reload the trained one
    try:
        h5save(f.name, lrn_)
    except Exception, e:
        raise AssertionError, \
              "Failed to store trained lrn due to %r" % (e,)

    # This lrn__ is doubly stored/loaded ;-)
    try:
        lrn__ = h5load(f.name)
    except Exception, e:
        raise AssertionError, \
              "Failed to load trained lrn due to %r" % (e,)

    # Verify that we have the same ca enabled
    # TODO
    #ok_(set(lrn.ca.enabled) == set(lrn__.ca.enabled))
    # and having the same values?
    # TODO

    # now lets do predict and manually compute error
    predictions = lrn__.predict(ds[ds.sa.train == 2].samples)
    error__ = errorfx(predictions, ds[ds.sa.train == 2].sa.targets)

    if 'non-deterministic' in lrn_.__tags__:
        # might be different... let's allow to vary quite a bit
        # and new error should be no more than twice the old one
        # (better than no check at all)
        # TODO: smarter check, since 'twice' is quite coarse
        #       especially if original error happens to be 0 ;)
        if cfg.getboolean('tests', 'labile', default='yes'):
            ok_(np.asscalar(error_) <= 2*np.asscalar(error))
            ok_(np.asscalar(error__) <= 2*np.asscalar(error))
    else:
        # must match precisely
        assert_array_equal(error, error_)
        assert_array_equal(error, error__)

    # TODO: verify ca's

    #print "I PASSED!!!! %s" % lrn
