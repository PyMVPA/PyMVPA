### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Wrap the libsvm package into a very simple class interface.
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from classifier import *
from mvpa import libsvm
import numpy

class SVM(Classifier):
    """ Support Vector Machine Classifier.
    
    This is a simple interface to the libSVM package.
    """
    def __init__(self, data, **kwargs):
        # init base class
        Classifier.__init__(self, data, ['feature_benchmark'] )

        # check if there is a libsvm version with configurable
        # noise reduction ;)
        if hasattr(libsvm.svmc, 'svm_set_verbosity'):
            libsvm.svmc.svm_set_verbosity( 0 )

        self.param = libsvm.svm_parameter( **(kwargs) )

        # libsvm needs doubles
        if data.pattern.dtype == 'float64':
            src = data.pattern
        else:
            src = data.pattern.astype('double')

        self.data = libsvm.svm_problem( data.reg.tolist(), src )

        self.train()


    def train(self):
        self.model = libsvm.svm_model( self.data, self.param)


    def predict(self, data):
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')
        return [ self.model.predict( p ) for p in src ]


    def getFeatureBenchmark(self):
        return self.model.get_feature_benchmark()

