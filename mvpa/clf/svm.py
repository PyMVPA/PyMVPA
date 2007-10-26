### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Wrap the libsvm package into a very simple class interface.
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from mvpa.clf.classifier import Classifier
from mvpa.clf import libsvm

class SVM(Classifier):
    """ Support Vector Machine Classifier.

    This is a simple interface to the libSVM package.
    """
    _param = [ 'eps' ] + Classifier._param

    def __init__(self, **kwargs):
        # init base class
        Classifier.__init__(self, ['feature_benchmark'] )

        # check if there is a libsvm version with configurable
        # noise reduction ;)
        if hasattr(libsvm.svmc, 'svm_set_verbosity'):
            libsvm.svmc.svm_set_verbosity( 0 )

        self.param = libsvm.svm_parameter( **(kwargs) )


    def train(self, data):
        # libsvm needs doubles
        if data.samples.dtype == 'float64':
            src = data.samples
        else:
            src = data.samples.astype('double')

        svmprob = libsvm.svm_problem( data.labels.tolist(), src )

        self.model = libsvm.svm_model( svmprob, self.param)


    def predict(self, data):
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')
        return [ self.model.predict( p ) for p in src ]


    def getFeatureBenchmark(self):
        return self.model.get_feature_benchmark()

