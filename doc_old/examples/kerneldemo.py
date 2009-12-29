#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Kernel-Demo
===========

This is an example demonstrating various kernel implementation in PyMVPA.
"""

import numpy as N
import pylab as P

#from mvpa.suite import *
from mvpa.base import cfg
from mvpa.kernels.np import *


# N.random.seed(1)
data = N.random.rand(4, 2)

for kernel_class, kernel_args in (
    (ConstantKernel, {'sigma_0':1.0}),
    (ConstantKernel, {'sigma_0':1.0}),
    (GeneralizedLinearKernel, {'Sigma_p':N.eye(data.shape[1])}),
    (GeneralizedLinearKernel, {'Sigma_p':N.ones(data.shape[1])}),
    (GeneralizedLinearKernel, {'Sigma_p':2.0}),
    (GeneralizedLinearKernel, {}),
    (ExponentialKernel, {}),
    (SquaredExponentialKernel, {}),
    (Matern_3_2Kernel, {}),
    (Matern_5_2Kernel, {}),
    (RationalQuadraticKernel, {}),
    ):
    kernel = kernel_class(**kernel_args)
    print kernel
    result = kernel.compute(data)

# In the following we draw some 2D functions at random from the
# distribution N(O,kernel) defined by each available kernel and
# plot them. These plots shows the flexibility of a given kernel
# (with default parameters) when doing interpolation. The choice
# of a kernel defines a prior probability over the function space
# used for regression/classfication with GPR/GPC.
count = 1
for k in kernel_dictionary.keys():
    P.subplot(3, 4, count)
    # X = N.random.rand(size)*12.0-6.0
    # X.sort()
    X = N.arange(-1, 1, .02)
    X = X[:, N.newaxis]
    ker = kernel_dictionary[k]()
    ker.compute(X, X)
    print k
    K = N.asarray(ker)
    for i in range(10):
        f = N.random.multivariate_normal(N.zeros(X.shape[0]), K)
        P.plot(X[:, 0], f, "b-")

    P.title(k)
    P.axis('tight')
    count += 1

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()
