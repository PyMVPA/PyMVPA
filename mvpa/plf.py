### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Wrap the python libsvm package into a very simple class interface.
#
#    Copyright (C) 2007 by
#    Ingo Fruend <ingo.fruend@gmail.com>
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

import sys
import numpy as N

class IterationError(Exception):
    pass

class PLF(Classifier):
    def __init__(self,lm=1,criterion=1,reduce=False,maxiter=20,verbose=False):
        """
        Initialize a penalized logistic regression analysis

        Input:
        =====

        lm     the penalty term lambda
        criterion is the criterion applied to judge convergence
        if reduce is not False, the rank of the data is reduced before
               performing the calculations. In that case, reduce is taken as
               the fraction of the first singular value, at which a dimension
               is not considered significant anymore. A reasonable criterion
               is reduce=0.01
        maxiter maximum number of iterations. If no convergence occurs after
               this number of iterations, an exception is raised
        verbose switches on/off logging information to stderr
        """
        self.__lm   = lm
        self.__criterion = criterion
        self.__reduce = reduce
        self.__maxiter = maxiter
        self.__verbose = verbose


    def train(self,data):
        """
        data   is a MVPApattern object containing the data
        """
        # Set up the environment for fitting the data
        X = data.pattern.T
        d = data.reg
        if not list(set(d))==[0,1]:
            raise ValueError, "Regressors for logistic regression should be [0,1]"

        if self.__reduce:
            # Data have reduced rank
            from scipy.linalg import svd

            # Compensate for reduced rank:
            # Select only the n largest eigenvectors
            U,S,V = svd(X.T)
            S /= S[0]
            V = N.matrix(V[:,:N.max(N.where(S>self.__reduce))+1])
            X = (X.T*V).T # Map Data to the subspace spanned by the eigenvectors

        nfeatures,npatterns = X.shape

        # Weighting vector
        w  = N.matrix(N.zeros( (nfeatures+1,1),'d'))
        # Error for convergence criterion
        dw = N.matrix(N.ones(  (nfeatures+1,1),'d'))
        # Patterns of interest in the columns
        X = N.matrix(\
                N.concatenate((X,N.ones((1,npatterns),'d')),0)\
                )
        p = N.matrix(N.zeros((1,npatterns),'d'))
        # Matrix implementation of penalty term
        Lambda = self.__lm * N.identity(nfeatures+1,'d')
        Lambda[nfeatures,nfeatures] = 0
        # Gradient
        g = N.matrix(N.zeros((nfeatures+1,1),'d'))
        # Fisher information matrix
        H = N.matrix(N.identity(nfeatures+1,'d'))

        # Optimize
        k = 0
        while N.sum(N.ravel(dw.A**2))>self.__criterion:
            p[:,:] = self.__f(w.T * X)
            g[:,:] = X * (d-p).T - Lambda * w
            H[:,:] = X * N.diag(p.A1 * (1-p.A1)) * X.T + Lambda
            dw[:,:] = H.I * g
            w += dw
            k += 1
            if k>self.__maxiter:
                raise IterationError, "More than 20 Iterations without convergence"

        if self.__verbose:
            sys.stderr.write(\
                    "PLF converged after %d steps\nError: %g\n" %\
                    (k,N.sum(N.ravel(dw.A**2))))

        if self.__reduce:
            # We have computed in rank reduced space ->
            # Project to original space
            self.w = V*w[:-1]
            self.offset = w[-1]
        else:
            self.w = w[:-1]
            self.offset = w[-1]

    def __f(self,y):
        """This is the logistic function f, that is used for determination of
        the vector w"""
        return 1./(1+N.exp(-y))

    def predict(self,data):
        """
        Predict the class labels for the provided data

        Returns a list of class labels
        """
        data = N.matrix(N.array(data))
        return N.ravel(self.__f(self.offset+data*self.w) > 0.5)

