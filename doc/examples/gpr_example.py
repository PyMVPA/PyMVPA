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
Run Gaussian Process Regression (GPR) on a simple 1D example
using squared exponential (i.e., Gaussian or RBF) kernel and
different hyperparameters.
"""


from mvpa.suite import *
import pylab as P


def compute_prediction(sigma_f, length_scale, sigma_noise, 
                       regression, dataset, data_test, label_test, F,
                       logml=True):
    """Given hyperparameters and data (train and test), compute GPR,
    plot the predictions on test data (if possible) and estimate error
    rate.
    """

    data_train = dataset.samples
    label_train = dataset.labels
    kse = KernelSquaredExponential(length_scale=length_scale,
                                   sigma_f=sigma_f)
    g = GPR(kse, sigma_noise=sigma_noise, regression=regression)
    print g
    if regression:
        g.states.enable("predicted_variances")
        pass

    if logml:
        g.states.enable("log_marginal_likelihood")
        pass

    g.train(dataset)
    prediction = g.predict(data_test)

    # print label_test
    # print prediction
    accuracy = None
    if regression:
        accuracy = N.sqrt(((prediction-label_test)**2).sum()/prediction.size)
        print "RMSE:", accuracy
    else:
        accuracy = (prediction.astype('l')==label_test.astype('l')).sum() \
                   / float(prediction.size)
        print "accuracy:", accuracy
        pass

    if F == 1:
        P.title("GPR: sigma_f=%0.2f , length_s=%0.2f , sigma_n=%0.2f" % (sigma_f,length_scale,sigma_noise))
        P.plot(data_train, label_train, "ro", label="train")
        P.plot(data_test, prediction, "b-", label="prediction")
        P.plot(data_test, label_test, "g+", label="test")
        if regression:
            P.plot(data_test, prediction-N.sqrt(g.predicted_variances),
                       "b--", label=None)
            P.plot(data_test, prediction+N.sqrt(g.predicted_variances),
                       "b--", label=None)
            P.text(0.5, -0.8, "RMSE="+"%.3f" %(accuracy))
            P.text(0.5, -0.95, "LMLtest="+"%.3f" %(g.log_marginal_likelihood))
        else:
            P.text(0.5, -0.8, "accuracy="+str(accuracy))
            pass
        P.legend(loc='lower right')
        pass

    print "LMLtest:", g.log_marginal_likelihood


if __name__=="__main__":

    # Generate dataset for training:
    train_size = 40
    F = 1
    dataset = data_generators.sinModulated(train_size, F)

    # Generate dataset for testing:
    test_size = 100
    dataset_test = data_generators.sinModulated(test_size, F, flat=True)
    data_test = dataset_test.samples
    label_test = dataset_test.labels

    # Hyperparameters. Each row is [sigma_f, length_scale, sigma_noise]
    hyperparameters = N.array([[1.0, 0.2, 0.4],
                               [1.0, 0.1, 0.1],
                               [1.0, 1.0, 0.1],
                               [1.0, 0.1, 1.0]])

    rows = 2
    columns = 2
    P.figure()
    for i in range(rows*columns):
        P.subplot(rows,columns,i+1)
        # Plot some graph and figures using the hyperparameters just computed.
        sigma_f, length_scale, sigma_noise = hyperparameters[i,:]
        compute_prediction(sigma_f, length_scale, \
                           sigma_noise, True, \
                           dataset,data_test, label_test, F)
        pass

    P.show()
