#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Gaussian Process Regression (GPR)."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.clfs.classifier import Classifier

from scipy import weave
from scipy.weave import converters
import time

def distance_matrix(data1,data2,symmetric=False):
    """
    Compute (euclidean) distance matrix between two datasets. This
    implementation is super fast since it uses inline C code.    

    Note that when symmetric==True only half matrix is computed and
    then replicated (optimization).

    Note that the resulting distance matrix has data1 on rows and
    data2 on columns.
    I believe it's more readable. All the code inthe following
    functions is adapted to this new convention.
    """
    size1 = data1.shape[0]
    size2 = data2.shape[0]
    F = data1.shape[1]
    dm = N.zeros((data1.shape[0],data2.shape[0]),'d')
    code = None
    if symmetric==False:
        code = """
        int i,j,t;
        double tmp,distance;
        for (i=0;i<size1;i++) {
            for (j=0;j<size2;j++) {
                tmp = 0.0;
                for(t=0;t<F;t++) {
                    distance = data1(i,t)-data2(j,t);
                    tmp = tmp+distance*distance;
                    }
                dm(i,j) = tmp;
                }
            }
        return_val = 0;
        """
    else:
        code = """
        int i,j,t;
        double tmp,distance;
        for (i=0;i<size1-1;i++) {
            for (j=i;j<size2;j++) {
                tmp = 0.0;
                for(t=0;t<F;t++) {
                    distance = data1(i,t)-data2(j,t);
                    tmp = tmp+distance*distance;
                    }
                dm(i,j) = tmp;
                }
            }
        return_val = 0;
        """        
    t = time.time()
    retval = weave.inline(code,
                           ['data1','size1','data2','size2','F','dm'],
                           type_converters=converters.blitz,
                           compiler = 'gcc')
    if symmetric==True:
        dm = dm+N.triu(dm).T # copy upper part to lower part
        pass
    print "Distance matrix computed in",time.time()-t,"sec."
    
    return dm



class Kernel(object):
    """Kernel function base class.
       
    """
    def __repr__(self):
        return "A basic Kernel."

    pass



class SquaredExponential(Kernel):
    """The Squared Exponential kernel function class.

    """
    def __init__(self,length_scale=0.01,sigma_noise=0.001, **kwargs):
        """Initialize the Squared Exponential class.

        :Parameters:
          length_scale : float
            the characteristic lengthscale of the phenomenon under investigation.
            (Defaults to 0.01)

          sigma_noise : float
            the standard deviation of the gaussian noise.
            (Defaults to 0.001)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)
        
        self.length_scale = length_scale
        self.sigma_noise = sigma_noise
        return

    def __repr__(self):
        return "Squared_Exponential_kernel(length_scale=%f,sigma_noise=%f)" % (self.length_scale,self.sigma_noise)

    def compute(self,distance_matrix):
        return N.exp(-distance_matrix/(2.0*self.length_scale**2))

    pass



class GPR(Classifier):
    """Gaussian Process Regression (GPR).
    
    """

    def __init__(self, kernel=SquaredExponential(lengthscale=0.01,sigma_noise=0.001), **kwargs):
        """
        Initialize a GPR regression analysis.

        :Parameters:
          kernel : Kernel
            a kernel object defining the covariance between instances.
            (Default to SquaredExponential(lengthscale=0.01,sigma_noise=0.001))

        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint happiness
        self.w = None

        # It does not make sense to calculate a confusion matrix for a GPR
        self.states.enable('training_confusion', False)

        # set kernel
        self.__kernel = kernel
        return
    


    def __repr__(self):
        """String summary of the object
        """
        return """GPR(kernel=%s, enabled_states=%s)""" % \
               (self.__kernel, str(self.states.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        print "Computing train train Distance Matrix"
        self.train_fv = data.samples
        self.train_labels = data.labels
        self.dm_train_train = distance_matrix(self.train_fv,self.train_fv,symmetric=True)
        self.km_train_train = self.__kernel.compute(self.dm_train_train)

        self.L = N.linalg.cholesky(self.km_train_train+self.__kernel.sigma_noise**2*N.identity(self.km_train_train.shape[0],'d'))
        self.alpha = N.linalg.solve(self.L.transpose(),N.linalg.solve(self.L,self.train_labels))

        return


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """

        print "Computing test train Distance Matrix"
        dm_train_test = distance_matrix(self.train_fv,data,symmetric=False)
        print "Computing test test Distance Matrix"
        dm_test_test = distance_matrix(data,data,symmetric=True)

        km_train_test = self.__kernel.compute(dm_train_test)
        km_test_test = self.__kernel.compute(dm_test_test)

        predicted_label = N.dot(km_train_test.transpose(),self.alpha)
        # v = N.linalg.solve(L,km_train_test)
        # predicted_variance = N.diag(km_test_test-N.dot(v.transpose(),v))
        # log_marginal_likelihood = -0.5*N.dot(train_label,alpha)-N.log(L.diagonal()).sum()-km_train_train.shape[0]*0.5*N.log(2*N.pi)
        

        return predicted_label

    pass


if __name__=="__main__":

    
    k = Kernel()
    print k

    kse = SquaredExponential()
    print kse
    
    g = GPR()
    print g
    
