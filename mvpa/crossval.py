### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Generic Cross-Validation
#
#    Copyright (C) 2006-2007 by
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

import numpy as N
import support



class ErrorFunction(object):
    """
    Dummy error function
    """
    pass


class RMSEFunction(ErrorFunction):
    """
    """
    def __call__(self, predicted, desired):
        difference = predicted - desired
        return sqrt(N.dot(difference, difference))



class SplitProcessing(object):
    """
    Base/dummy class
    """
    def __call__(self, splitter, split, classifier, predictions ):
        raise NotImplementedError



class RMSESplitProcessing(SplitProcessing):
    def __call__(self, splitter, split, classifier, predictions ):
        RMSEFunction error;
        return error( predictions, split[1].regs)




class CrossValidation(object):
    """
    """
    def __init__( self,
                  splitter,
                  classifier,
                  splitprocessing ):
        """

        @postprocessing  --- list of instances which gets arguments:
                generated split, splitter object, classifier

        """
        pass


    def __call__( self, dataset ):
        """

        Returns a sequence of postprocessingResults.
        """
        # classifier.predict(split[1].samples)
        pass


###############
##           ##
## Old stuff ##
##           ##
###############



class CrossValidation( object ):
    """ Generic N-M cross-validation with arbitrary classification
    algorithms.
    """
    def __init__( self,
                  pattern,
                  classifier,
                  clfcallback = None,
                  **kwargs ):
        """
        Initialize the cross-validation.

          pattern:    MVPAPattern object containing the data, classification
                      targets and origins for the cross-validation.
          classifier: A classifier instance that shall be used to do the actual
                      classification.
          clfcallback:
                      a callable that is called at the end of each
                      cross-validation fold with the trained classifier as the
                      only argument.

        Additional keyword arguments are passed to the CrossvalPatternGenerator
        object (see its documentation for more info).
        """
        # setup pattern generation for xval
        self.__cvpg = xvalpattern.CrossvalPatternGenerator(pattern, **(kwargs))

        self.setClassifier( classifier )
        self.setClassifierCallback( clfcallback )

        self.__perf = []
        self.__contingency_tbl = None
        self.__test_samplelog = []
        self.__train_samplelog = []


    def __clean_logdata( self ):
        """ The internal method is called by run() to clean any previous log
        data.
        """
        self.__perf = []
        self.__contingency_tbl = None
        self.__test_samplelog = []
        self.__train_samplelog = []


    def setClassifier( self, classifier ):
        """ Set a classifier instance that is used to perform the
        classification.
        """
        if not hasattr( classifier, 'predict' ):
            raise ValueError, "Classifier object has to provide a " \
                         "'predict()' method."

        self.__clf = classifier


    def setClassifierCallback( self, callable ):

        """ Sets a callable that is called at the end of each cross-validation
        fold with the trained classifier as the only argument.
        """
        self.__clf_callback = callable


    def __call__( self, permutate = False ):
        """ Perform cross-validation.

        Parameters:
          permutate:  If True the regressor vector is permutated for each
                      cross-validation fold. In conjunction with ncvfoldsamples
                      this might be useful to test a classification algorithm
                      whether it can classify any noise ;)

        Returns:
          List of performance values (fraction of correct classifications) for
          each  cross-validation fold. More performance values are available
          via several class attributes.
        """
        # clean previous log data
        self.__clean_logdata()

        for train_samples, \
            train_samplesize, \
            test_samples, \
            test_samplesize in self.__cvpg(permutate):
                # log the sizes
                self.trainsamplelog.append( train_samplesize )
                self.testsamplelog.append( test_samplesize )

                clf, predictions = \
                    CrossValidation.getClassification( self.__clf,
                                                       train_samples,
                                                       test_samples )

                perf = ( predictions == test_samples.regs )

                # call classifier callback if any
                if not self.__clf_callback == None:
                    self.__clf_callback( clf )

                # store performance
                self.perf.append( perf.mean() )

                # store more details performance description
                # as contingency table
                self.__updateContingencyTbl(
                    support.buildConfusionMatrix( 
                        self.xvalpattern.pattern.reglabels,
                        test_samples.regs,
                        predictions ) )

        return self.perf


    @staticmethod
    def getClassification( clf, train, test ):
        """ Perform classification and get classifier predictions.

        Parameters:
            clf    - A classifier instance that shall be used to do the actual
                     classification.
            train  - MVPAPattern object with the training data.
            test   - MVPAPattern object with the test data

        Returns:
            - the trained classifier
            - a sequence with the classifier predictions
        """
        # train the classifier
        clf.train(train)

        # test
        predictions = N.array(
                        clf.predict(test.samples) )

        return clf, predictions


    def __updateContingencyTbl( self, tbl ):
        """ Internal method to store the sum of all contingency tables from
        all CV runs.
        """
        if self.__contingency_tbl == None:
            self.__contingency_tbl = tbl
        else:
            self.__contingency_tbl += tbl



    # read only props
    perf = property( fget=lambda self: self.__perf )
    clf  = property( fget=lambda self: self.__clf,
                     fset=setClassifier )
    xvalpattern    = property( fget=lambda self: self.__cvpg )
    testsamplelog  = property( fget=lambda self: self.__test_samplelog )
    trainsamplelog = property( fget=lambda self: self.__train_samplelog )
    contingencytbl = property( fget=lambda self: self.__contingency_tbl )
