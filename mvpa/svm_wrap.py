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

import svm
import numpy

class SVM:
    def __init__(self, data, **kwargs):
        self.param = svm.svm_parameter( **(kwargs) )

        self.data = svm.svm_problem(data.reg.tolist(), data.pattern)

        self.train()

    def train(self):
        self.model = svm.svm_model( self.data, self.param)

    def predict(self, data):
        return [ self.model.predict( p ) for p in data ]


