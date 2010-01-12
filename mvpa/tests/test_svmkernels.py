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
from mvpa.datasets import Dataset
from mvpa.clfs.libsvmc import SVM as lsSVM
from mvpa.clfs.sg import SVM as sgSVM

from tests_warehouse import datasets, sweepargs

class SVMKernelTests(unittest.TestCase):
    
    @sweepargs(clf=[lsSVM(), sgSVM()])
    def testBasicClfTrainPredict(self, clf):
        d = datasets['uni4medium']
        clf.train(d)
        clf.predict(d)
        pass

    def testCacheSpeedup(self):
        from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
        from mvpa.datasets.splitters import NFoldSplitter
        from time import time
        from mvpa.clfs.transerror import TransferError
        from mvpa.kernels.base import CachedKernel
        from mvpa.kernels.sg import RbfSGKernel
        from mvpa.misc.data_generators import normalFeatureDataset
        
        ck = sgSVM(kernel=CachedKernel(kernel=RbfSGKernel(sigma=2)), C=1)
        sk = sgSVM(kernel=RbfSGKernel(sigma=2), C=1)
        
        cv_c = CrossValidatedTransferError(TransferError(ck),
                                           splitter=NFoldSplitter())
        cv_s = CrossValidatedTransferError(TransferError(sk), 
                                           splitter=NFoldSplitter())
        
        #data = datasets['uni4large']
        P = 5000
        data = normalFeatureDataset(snr=2, perlabel=200, nchunks=10, 
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

