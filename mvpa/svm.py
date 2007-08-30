### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Wrap the python libsvm package into a very simple class interface.
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
from mvpa import svm
import numpy

class SVM(Classifier):
    def __init__(self, data, **kwargs):
        # init base class
        Classifier.__init__(self, data, [] )

        # check if there is a libsvm version with configurable
        # noise reduction ;)
        if hasattr(svm.svmc, 'svm_set_verbosity'):
            svm.svmc.svm_set_verbosity( 0 )

        self.param = svm.svm_parameter( **(kwargs) )

        # libsvm needs doubles
        if data.pattern.dtype == 'float64':
            src = data.pattern
        else:
            src = data.pattern.astype('double')

        self.data = svm.svm_problem( data.reg.tolist(), src )

        self.train()

    def train(self):
        self.model = svm.svm_model( self.data, self.param)

    def predict(self, data):
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')
        return [ self.model.predict( p ) for p in src ]


