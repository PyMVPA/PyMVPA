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

import numpy as np
from mvpa2.support.pylab import pl

#from mvpa2.suite import *
from mvpa2.base import cfg
from mvpa2.kernels.np import *


# np.random.seed(1)
data = np.random.rand(4, 2)

for kernel_class, kernel_args in (
    (ConstantKernel, {'sigma_0':1.0}),
    (ConstantKernel, {'sigma_0':1.0}),
    (GeneralizedLinearKernel, {'Sigma_p':np.eye(data.shape[1])}),
    (GeneralizedLinearKernel, {'Sigma_p':np.ones(data.shape[1])}),
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
    pl.subplot(3, 4, count)
    # X = np.random.rand(size)*12.0-6.0
    # X.sort()
    X = np.arange(-1, 1, .02)
    X = X[:, np.newaxis]
    ker = kernel_dictionary[k]()
    ker.compute(X, X)
    print k
    K = np.asarray(ker)
    for i in range(10):
        f = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
        pl.plot(X[:, 0], f, "b-")

    pl.title(k)
    pl.axis('tight')
    count += 1

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()
