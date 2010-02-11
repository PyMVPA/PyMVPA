# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for new Kernel-based SVMs"""

import unittest
import numpy as N

from mvpa.clfs.libsvmc import SVM as lsSVM
from mvpa.clfs.sg import SVM as sgSVM

from mvpa.testing import *
from mvpa.testing.datasets import datasets

class SVMKernelTests(unittest.TestCase):

    @sweepargs(clf=[lsSVM(), sgSVM()])
    def test_basic_clf_train_predict(self, clf):
        d = datasets['uni4medium']
        clf.train(d)
        clf.predict(d)
        pass

    def test_cache_speedup(self):
        from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
        from mvpa.datasets.splitters import NFoldSplitter
        from time import time
        from mvpa.clfs.transerror import TransferError
        from mvpa.kernels.base import CachedKernel
        from mvpa.kernels.sg import RbfSGKernel
        from mvpa.misc.data_generators import normal_feature_dataset

        ck = sgSVM(kernel=CachedKernel(kernel=RbfSGKernel(sigma=2)), C=1)
        sk = sgSVM(kernel=RbfSGKernel(sigma=2), C=1)

        cv_c = CrossValidatedTransferError(TransferError(ck),
                                           splitter=NFoldSplitter())
        cv_s = CrossValidatedTransferError(TransferError(sk),
                                           splitter=NFoldSplitter())

        #data = datasets['uni4large']
        P = 5000
        data = normal_feature_dataset(snr=2, perlabel=200, nchunks=10,
                                    means=N.random.randn(2, P), nfeatures=P)

        t0 = time()
        ck.params.kernel.compute(data)
        cachetime = time()-t0

        t0 = time()
        cached_err = cv_c(data)
        ccv_time = time()-t0

        t0 = time()
        norm_err = cv_s(data)
        ncv_time = time()-t0

        assert_almost_equal(N.asanyarray(cached_err),
                            N.asanyarray(norm_err))
        ok_(cachetime<ncv_time)
        ok_(ccv_time<ncv_time)
        #print 'Regular CV time: %s seconds'%ncv_time
        #print 'Caching time: %s seconds'%cachetime
        #print 'Cached CV time: %s seconds'%ccv_time

        speedup = ncv_time/(ccv_time+cachetime)
        #print 'Speedup factor: %s'%speedup

        # Speedup ideally should be 10, though it's not purely linear
        self.failIf(speedup < 2, 'Problem caching data - too slow!')


def suite():
    return unittest.makeSuite(KernelTests)


if __name__ == '__main__':
    import runner

