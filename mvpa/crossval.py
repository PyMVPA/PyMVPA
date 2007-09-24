### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: General N-M cross-validation and utility functions
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

import numpy
import support
import xvalpattern

class CrossValidation( object ):
    """ Generic N-M cross-validation with arbitrary classification
    algorithms.
    """
    def __init__( self,
                  pattern,
                  trainingsamples = None,
                  testsamples = None,
                  ncvfoldsamples = 1,
                  clfcallback = None ):
        """
        Initialize the cross-validation.

          pattern:    MVPAPattern object containing the data, classification
                      targets and origins for the cross-validation.
          trainingsamples:
                      Number of training patterns to be used in each
                      cross-validation fold. Please see the
                      setTrainingPatternSamples() method for special arguments.
          testsamples:
                      Number of test pattern to be used in each
                      cross-validation fold. Please see the
                      setTestPatternSamples() method for special arguments.
          ncvfoldsamples:
                      Number of time each cross-validation fold is run. This
                      is mostly usefull if a subset of the available patterns
                      is used for classification and the subset is randomly
                      selected for each CV-fold run (see the trainingsamples
                      and testsamples arguments).
          clfcallback:
                      a callable that is called at the end of each
                      cross-validation fold with the trained classifier as the
                      only argument.
        """
        # setup pattern generation for xval
        self.__cvpg = xvalpattern.CrossvalPatternGenerator(pattern)
        self.__cvpg.setTrainingPatternSamples( trainingsamples )
        self.__cvpg.setTestPatternSamples( testsamples )
        self.__cvpg.setNCVFoldSamples( ncvfoldsamples )

        self.__perf = []
        self.__contingency_tbl = None
        self.__test_samplelog = []
        self.__train_samplelog = []


        self.setClassifierCallback( clfcallback )


    def setClassifierCallback( self, callable ):
        """ Sets a callable that is called at the end of each cross-validation
        fold with the trained classifier as the only argument.
        """
        self.__clf_callback = callable


    def __clean_logdata( self ):
        """ The internal method is called by run() to clean any previous log
        data.
        """
        self.__perf = []
        self.__contingency_tbl = None
        self.__test_samplelog = []
        self.__train_samplelog = []


    def run( self, classifier, cvtype = 1, permutate = False, **kwargs ):
        """ Perform cross-validation.

        Parameters:
          permutate:  If True the regressor vector is permutated for each
                      cross-validation fold. In conjunction with ncvfoldsamples
                      this might be useful to test a classification algorithm
                      whether it can classify any noise ;)
          classifier: A class that shall be used the actual classification. Its
                      constructor must not have more than two required
                      arguments (data and regs - in this order).
                      The classifier has to train itself when creating the
                      classifier object!
          cvtype:     Type of cross-validation: N-cvtype
          **kwargs:   All keyword arguments are passed to the classifiers
                      constructor.

        Returns:
          List of performance values (fraction of correct classifications) for
          each  cross-validation fold. More performance values are available
          via several class attributes.
        """
        if not hasattr( classifier, 'predict' ):
            raise ValueError, "Classifier object has to provide a " \
                         "'predict()' method."

        # clean previous log data
        self.__clean_logdata()

        for train_samples, \
            train_samplesize, \
            test_samples, \
            test_samplesize in self.__cvpg(cvtype, permutate):
                # log the sizes
                self.trainsamplelog.append( train_samplesize )
                self.testsamplelog.append( test_samplesize )

                clf, predictions = \
                    CrossValidation.getClassification( classifier,
                                                       train_samples,
                                                       test_samples,
                                                       **(kwargs) )

                perf = ( predictions == test_samples.reg )

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
                        test_samples.reg,
                        predictions ) )

        return self.perf


    @staticmethod
    def getClassification( clf, train, test, **kwargs ):
        """ Perform classification and get classifier predictions.

        Parameters:
            clf    - A class that shall be used the actual classification. Its
                     constructor must not have more than two required
                     arguments (data and regs - in this order).
                     The classifier has to train itself when creating the
                     classifier object!
            train  - MVPAPattern object with the training data.
            test   - MVPAPattern object with the test data
            ...    - all additional keyword arguments are passed to the
                     classifier

        Returns:
            - the trained classifier
            - a sequence with the classifier predictions
        """
        # create classifier (must include training if necessary)
        clf = clf(train, **(kwargs) )

        # test
        predictions = numpy.array(
                        clf.predict(test.pattern) )

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
    xvalpattern = property( fget=lambda self: self.__cvpg )
    testsamplelog = property( fget=lambda self: self.__test_samplelog )
    trainsamplelog = property( fget=lambda self: self.__train_samplelog )
    contingencytbl = property( fget=lambda self: self.__contingency_tbl )
