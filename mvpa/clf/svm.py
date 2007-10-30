#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Wrap the libsvm package into a very simple class interface."""

from mvpa.clf.classifier import Classifier
from mvpa.clf.libsvm.svm import *

class SVM(Classifier):
    """ Support Vector Machine Classifier.

    This is a simple interface to the libSVM package.
    """
#    _param = [ 'eps' ] + Classifier._param

    def __init__(self, **kwargs):
        # init base class
        Classifier.__init__(self)

        # check if there is a libsvm version with configurable
        # noise reduction ;)
        if hasattr(svmc, 'svm_set_verbosity'):
            svmc.svm_set_verbosity( 0 )

        self.param = svm_parameter( **(kwargs) )


    def train(self, data):
        # libsvm needs doubles
        if data.samples.dtype == 'float64':
            src = data.samples
        else:
            src = data.samples.astype('double')

        svmprob = svm_problem( data.labels.tolist(), src )

        self.model = svm_model( svmprob, self.param)


    def predict(self, data):
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')
        return [ self.model.predict( p ) for p in src ]


    def getFeatureBenchmark(self):
        return self.model.get_feature_benchmark()

