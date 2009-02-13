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

from mvpa.suite import *
from mvpa.clfs.kernel import *
import pylab as P


# N.random.seed(1)
data = N.random.rand(4, 2)

for kernel_class, kernel_args in (
    (KernelConstant, {'sigma_0':1.0}),
    (KernelConstant, {'sigma_0':1.0}),
    (KernelLinear, {'Sigma_p':N.eye(data.shape[1])}),
    (KernelLinear, {'Sigma_p':N.ones(data.shape[1])}),
    (KernelLinear, {'Sigma_p':2.0}),
    (KernelLinear, {}),
    (KernelExponential, {}),
    (KernelSquaredExponential, {}),
    (KernelMatern_3_2, {}),
    (KernelMatern_5_2, {}),
    (KernelRationalQuadratic, {}),
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
    P.subplot(3,4,count)
    # X = N.random.rand(size)*12.0-6.0
    # X.sort()
    X = N.arange(-1,1,.02)
    X = X[:,N.newaxis]
    ker = kernel_dictionary[k]()
    K = ker.compute(X,X)
    for i in range(10):
        f = N.random.multivariate_normal(N.zeros(X.shape[0]),K)
        P.plot(X[:,0],f,"b-")

    P.title(k)
    P.axis('tight')
    count += 1

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()
