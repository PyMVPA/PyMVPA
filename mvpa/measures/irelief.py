#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FeaturewiseDatasetMeasure performing multivariate Iterative RELIEF
(I-RELIEF) algorithm.
See : Y. Sun, Iterative RELIEF for Feature Weighting: Algorithms, Theories,
and Applications, IEEE Trans. on Pattern Analysis and Machine Intelligence
(TPAMI), vol. 29, no. 6, pp. 1035-1051, June 2007."""


__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.measures.base import FeaturewiseDatasetMeasure
from mvpa.clfs.kernel import KernelSquaredExponential, KernelExponential, KernelMatern_3_2, KernelMatern_5_2

if __debug__:
    from mvpa.base import debug


class IterativeRelief(FeaturewiseDatasetMeasure):
    """`FeaturewiseDatasetMeasure` that performs multivariate I-RELIEF
    algorithm. Batch version.

    Batch I-RELIEF-2 feature weighting algorithm. Works for binary or
    multiclass class-labels. Batch version with complexity O(T*N^2*I),
    where T is the number of iterations, N the number of instances, I
    the number of features.

    See: Y. Sun, Iterative RELIEF for Feature Weighting: Algorithms,
    Theories, and Applications, IEEE Trans. on Pattern Analysis and
    Machine Intelligence (TPAMI), vol. 29, no. 6, pp. 1035-1051, June
    2007. http://plaza.ufl.edu/sunyijun/Paper/PAMI_1.pdf

    Note that current implementation allows to use only
    exponential-like kernels. Support for linear kernel will be
    added later.
    """
    def __init__(self, threshold = 1.0e-2, kernel = None, kernel_width = 1.0, w_guess = None, **kwargs):
        """Constructor of the IRELIEF class.

        """
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        self.threshold = threshold # Threshold in W changes (stopping criterion for irelief).
        if kernel == None:
            self.kernel = KernelExponential
        else:
            self.kernel = kernel
            pass
        self.w_guess = w_guess
        self.w = None
        self.kernel_width = kernel_width
        pass


    def compute_M_H(self,label):
        """Compute hit/miss dictionaries.

        For each instance compute the set of indices having the same
        class label and different class label.

        Note that this computation is independent of the number of
        features.
        """
        
        M = {}
        H = {}
        for i in range(label.size):
            M[i] = N.where(label!=label[i])[0]
            tmp = (N.where(label==label[i])[0]).tolist()
            tmp.remove(i)
            assert(tmp!=[]) # Assert that there are at least two exampls for class label[i]
            H[i] = N.array(tmp)
            pass
        return M,H
    
    
    def _call(self, dataset):
        """Computes featurewise I-RELIEF weights."""
        if self.w_guess==None:
            self.w = N.ones(dataset.samples.shape[1],'d')
            self.w = self.w/(self.w.sum()) # equal initial weights (sum up to 1.0)
        else:
            self.w = self.w_guess/self.w_guess.sum() # do normalization to be safe :)
            pass
        
        M, H = self.compute_M_H(dataset.labels)

        while True:
            self.k = self.kernel(length_scale = 1.0/(N.sqrt(self.w*self.kernel_width)))
            d_w_k = self.k.compute(dataset.samples)
            # set d_w_k to zero where distance=0 (i.e. kernel ==
            # 1.0), otherwise I-RELIEF could not converge.
            # XXX Note that kernel==1 for distance=0 only for
            # exponential kernels!!  IMPROVE
            d_w_k[N.abs(d_w_k-1.0)<1.0e-15] = 0.0
            ni = N.zeros(dataset.samples.shape[1],'d')
            for n in range(dataset.samples.shape[0]):
                gamma_n = 1.0 - N.nan_to_num(d_w_k[n,M[n]].sum() / (d_w_k[n,:].sum()-d_w_k[n,n]))
                alpha_n = N.nan_to_num(d_w_k[n,M[n]]/(d_w_k[n,M[n]].sum()))
                beta_n = N.nan_to_num(d_w_k[n,H[n]]/(d_w_k[n,H[n]].sum()))

                m_n = (N.abs(dataset.samples[n,:]-dataset.samples[M[n],:])*alpha_n[:,N.newaxis]).sum(0)
                h_n = (N.abs(dataset.samples[n,:]-dataset.samples[H[n],:])*beta_n[:,N.newaxis]).sum(0)
                ni += gamma_n*(m_n-h_n)
                pass
            ni = ni/dataset.samples.shape[0]
            
            ni_plus = N.clip(ni,0.0,N.inf) # set all negative elements to zero
            w_new = N.nan_to_num(ni_plus/(N.sqrt((ni_plus**2).sum())))
            change = N.abs(w_new-self.w).sum()
            # print "change=%.4f max=%f min=%.4f mean=%.4f std=%.4f #nan=%d" % (change,w_new.max(),w_new.min(),w_new.mean(),w_new.std(),N.isnan(w_new).sum())
            
            # update weights:
            self.w = w_new
            if change<self.threshold:
                break
            pass
        
        return self.w

