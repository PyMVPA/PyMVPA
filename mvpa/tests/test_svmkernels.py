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

from mvpa.generators.partition import NFoldPartitioner
from mvpa.measures.base import CrossValidation, TransferMeasure


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

        cv_c = CrossValidation(ck, NFoldPartitioner())
        cv_s = CrossValidation(sk, NFoldPartitioner())

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

        cvte = CrossValidation(clf, NFoldPartitioner())
        cvte_ = CrossValidation(clf_, NFoldPartitioner())

        splitter = Splitter('train')
        postproc=BinaryFxNode(mean_mismatch_error, 'targets')
        te = TransferMeasure(clf, splitter, postproc=postproc)
        te_ = TransferMeasure(clf_, splitter, postproc=postproc)

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

            raise RuntimeError('The following test could not be refactored, '
                    'because Shogun keeps core-dumping :-(')
            ssel = np.round(datasets['uni2large'].samples[:5, 0]).astype(int)
            terr = te(datasets['uni3small'][ssel], datasets['uni3small_train'][::2])
            terr_ = te_(datasets['uni3small_test'][ssel], datasets['uni3small_train'][::2])
            ok_(~ck._recomputed)
            ok_(terr == terr_)

    def test_vstack_and_origids_issue(self):
        # That is actually what swaroop hit
        skip_if_no_external('shogun', ver_dep='shogun:rev', min_version=4455)

        # Inspired by the problem Swaroop ran into
        k  = LinearSGKernel(normalizer_cls=False)
        k_ = LinearSGKernel(normalizer_cls=False)   # to be cached
        ck = CachedKernel(k_)

        clf = sgSVM(svm_impl='libsvm', kernel=k, C=-1)
        clf_ = sgSVM(svm_impl='libsvm', kernel=ck, C=-1)

        cvte = CrossValidation(clf, NFoldPartitioner())
        cvte_ = CrossValidation(clf_, NFoldPartitioner())

        ds = datasets['uni2large_test'].copy(deep=True)
        ok_(~('orig_ids' in ds.sa))     # assure that there are None
        ck.compute(ds)                  # so we initialize origids
        ok_('origids' in ds.sa)
        ds2 = ds.copy(deep=True)
        ds2.samples = np.zeros(ds2.shape)
        from mvpa.base.dataset import vstack
        ds_vstacked = vstack((ds2, ds))
        # should complaint now since there would not be unique
        # samples' origids
        if __debug__:
            assert_raises(ValueError, ck.compute, ds_vstacked)

        ds_vstacked.init_origids('samples')      # reset origids
        ck.compute(ds_vstacked)

        errs = cvte(ds_vstacked)
        errs_ = cvte_(ds_vstacked)
        # Following test would have failed since origids
        # were just ints, and then non-unique after vstack
        assert_array_equal(errs.samples, errs_.samples)

def suite():
    return unittest.makeSuite(SVMKernelTests)


if __name__ == '__main__':
    import runner

