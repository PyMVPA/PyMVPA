# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for new Kernel-based SVMs"""

import numpy as np
from time import time

from mvpa.testing import *
from mvpa.testing.datasets import datasets
skip_if_no_external('shogun')

from mvpa.kernels.base import CachedKernel
from mvpa.kernels.sg import RbfSGKernel, LinearSGKernel

from mvpa.misc.data_generators import normal_feature_dataset

from mvpa.clfs.libsvmc import SVM as lsSVM
from mvpa.clfs.sg import SVM as sgSVM

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError


class SVMKernelTests(unittest.TestCase):

    @sweepargs(clf=[lsSVM(), sgSVM()])
    def test_basic_clf_train_predict(self, clf):
        d = datasets['uni4medium']
        clf.train(d)
        clf.predict(d)
        pass

    def test_cache_speedup(self):
        skip_if_no_external('shogun', ver_dep='shogun:rev', min_version=4455)

        ck = sgSVM(kernel=CachedKernel(kernel=RbfSGKernel(sigma=2)), C=1)
        sk = sgSVM(kernel=RbfSGKernel(sigma=2), C=1)

        cv_c = CrossValidatedTransferError(TransferError(ck),
                                           splitter=NFoldSplitter())
        cv_s = CrossValidatedTransferError(TransferError(sk),
                                           splitter=NFoldSplitter())

        #data = datasets['uni4large']
        P = 5000
        data = normal_feature_dataset(snr=2, perlabel=200, nchunks=10,
                                    means=np.random.randn(2, P), nfeatures=P)

        t0 = time()
        ck.params.kernel.compute(data)
        cachetime = time()-t0

        t0 = time()
        cached_err = cv_c(data)
        ccv_time = time()-t0

        t0 = time()
        norm_err = cv_s(data)
        ncv_time = time()-t0

        assert_almost_equal(np.asanyarray(cached_err),
                            np.asanyarray(norm_err))
        ok_(cachetime<ncv_time)
        ok_(ccv_time<ncv_time)
        #print 'Regular CV time: %s seconds'%ncv_time
        #print 'Caching time: %s seconds'%cachetime
        #print 'Cached CV time: %s seconds'%ccv_time

        speedup = ncv_time/(ccv_time+cachetime)
        #print 'Speedup factor: %s'%speedup

        # Speedup ideally should be 10, though it's not purely linear
        self.failIf(speedup < 2, 'Problem caching data - too slow!')

    def test_cached_kernel_different_datasets(self):
        skip_if_no_external('shogun', ver_dep='shogun:rev', min_version=4455)

        # Inspired by the problem Swaroop ran into
        k  = LinearSGKernel(normalizer_cls=False)
        k_ = LinearSGKernel(normalizer_cls=False)   # to be cached
        ck = CachedKernel(k_)

        clf = sgSVM(svm_impl='libsvm', kernel=k, C=-1)
        clf_ = sgSVM(svm_impl='libsvm', kernel=ck, C=-1)

        cvte = CrossValidatedTransferError(
            TransferError(clf), NFoldSplitter())
        cvte_ = CrossValidatedTransferError(
            TransferError(clf_), NFoldSplitter())

        te = TransferError(clf)
        te_ = TransferError(clf_)

        for r in xrange(2):
            ds1 = datasets['uni2medium']
            errs1 = cvte(ds1)
            ck.compute(ds1)
            ok_(ck._recomputed)
            errs1_ = cvte_(ds1)
            ok_(~ck._recomputed)
            assert_array_equal(errs1, errs1_)

            ds2 = datasets['uni3small']
            errs2 = cvte(ds2)
            ck.compute(ds2)
            ok_(ck._recomputed)
            errs2_ = cvte_(ds2)
            ok_(~ck._recomputed)
            assert_array_equal(errs2, errs2_)

            ssel = np.round(datasets['uni2large'].samples[:5, 0]).astype(int)
            terr = te(datasets['uni3small_test'][ssel], datasets['uni3small_train'][::2])
            terr_ = te_(datasets['uni3small_test'][ssel], datasets['uni3small_train'][::2])
            ok_(~ck._recomputed)
            ok_(terr == terr_)

def suite():
    return unittest.makeSuite(SVMKernelTests)


if __name__ == '__main__':
    import runner

