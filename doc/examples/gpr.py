#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
The effect of different hyperparameters in GPR
==============================================

.. index:: GPR

The following example runs Gaussian Process Regression (GPR) on a
simple 1D dataset using squared exponential (i.e., Gaussian or RBF)
kernel and different hyperparameters. The resulting classifier
solutions are finally visualized in a single figure.

As usual we start by importing all of PyMVPA:
"""

# Lets use LaTeX for proper rendering of greek
from matplotlib import rc
rc('text', usetex=True)

from mvpa2.suite import *

"""
The next lines build two datasets using one of PyMVPA's data
generators.
"""

# Generate dataset for training:
train_size = 40
F = 1
dataset = data_generators.sin_modulated(train_size, F)

# Generate dataset for testing:
test_size = 100
dataset_test = data_generators.sin_modulated(test_size, F, flat=True)

"""
The last configuration step is the definition of four sets of
hyperparameters to be used for GPR.
"""

# Hyperparameters. Each row is [sigma_f, length_scale, sigma_noise]
hyperparameters = np.array([[1.0, 0.2, 0.4],
                           [1.0, 0.1, 0.1],
                           [1.0, 1.0, 0.1],
                           [1.0, 0.1, 1.0]])

"""
The plotting of the final figure and the actually GPR runs are
performed in a single loop.
"""

rows = 2
columns = 2
pl.figure(figsize=(12, 12))
for i in range(rows*columns):
    pl.subplot(rows, columns, i+1)
    regression = True
    logml = True

    data_train = dataset.samples
    label_train = dataset.sa.targets
    data_test = dataset_test.samples
    label_test = dataset_test.sa.targets

    """
    The next lines configure a squared exponential kernel with the set of
    hyperparameters for the current subplot and assign the kernel to the GPR
    instance.
    """

    sigma_f, length_scale, sigma_noise = hyperparameters[i, :]
    kse = SquaredExponentialKernel(length_scale=length_scale,
                                   sigma_f=sigma_f)
    g = GPR(kse, sigma_noise=sigma_noise)
    if not regression:
        g = RegressionAsClassifier(g)
    print g

    if regression:
        g.ca.enable("predicted_variances")

    if logml:
        g.ca.enable("log_marginal_likelihood")

    """
    After training GPR the predictions are queried by passing the test
    dataset samples and accuracy measures are computed.
    """

    g.train(dataset)
    prediction = g.predict(data_test)

    # print label_test
    # print prediction
    accuracy = None
    if regression:
        accuracy = np.sqrt(((prediction-label_test)**2).sum()/prediction.size)
        print "RMSE:", accuracy
    else:
        accuracy = (prediction.astype('l')==label_test.astype('l')).sum() \
                   / float(prediction.size)
        print "accuracy:", accuracy

    """
    The remaining code simply plots both training and test datasets, as
    well as the GPR solutions.
    """

    if F == 1:
        pl.title(r"$\sigma_f=%0.2f$, $length_s=%0.2f$, $\sigma_n=%0.2f$" \
                % (sigma_f,length_scale,sigma_noise))
        pl.plot(data_train, label_train, "ro", label="train")
        pl.plot(data_test, prediction, "b-", label="prediction")
        pl.plot(data_test, label_test, "g+", label="test")
        if regression:
            pl.plot(data_test, prediction - np.sqrt(g.ca.predicted_variances),
                       "b--", label=None)
            pl.plot(data_test, prediction+np.sqrt(g.ca.predicted_variances),
                       "b--", label=None)
            pl.text(0.5, -0.8, "$RMSE=%.3f$" %(accuracy))
            pl.text(0.5, -0.95, "$LML=%.3f$" %(g.ca.log_marginal_likelihood))
        else:
            pl.text(0.5, -0.8, "$accuracy=%s" % accuracy)

        pl.legend(loc='lower right')

    print "LML:", g.ca.log_marginal_likelihood


if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()
