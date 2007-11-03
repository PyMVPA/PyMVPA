#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Wrap the libsvm package into a very simple class interface."""

from mvpa.misc.param import Parameter
from mvpa.clf.classifier import Classifier
from mvpa.clf.libsvm.svm import *


class SVMBase(Classifier):
    """ Support Vector Machine Classifier.

    This is a simple interface to the libSVM package.
    """
    # init the parameter interface
    params = Classifier.params.copy()
    params['eps'] = Parameter(0.001,
                              min=0,
                              descr='tolerance of termination criterium')


    def __init__(self, **kwargs):
        # init base class
        Classifier.__init__(self)

        # check if there is a libsvm version with configurable
        # noise reduction ;)
        if hasattr(svmc, 'svm_set_verbosity'):
            svmc.svm_set_verbosity( 0 )

        self.param = SVMParameter( **(kwargs) )
        self.model = None


    def __repr__(self):
        """ String summary over the object
        """
        return """SVM:
         params: %s """ % (self.param)


    def train(self, data):
        """Train SVM
        """
        # libsvm needs doubles
        if data.samples.dtype == 'float64':
            src = data.samples
        else:
            src = data.samples.astype('double')

        svmprob = SVMProblem( data.labels.tolist(), src )

        self.model = SVMModel( svmprob, self.param)


    def predict(self, data):
        """Predict values for the data
        """
        # libsvm needs doubles
        if data.dtype == 'float64':
            src = data
        else:
            src = data.astype('double')
        return [ self.model.predict( p ) for p in src ]


#    def getFeatureBenchmark(self):
#        """XXX Do we need this one?
#        """
#        return self.model.getFeatureBenchmark()
#
