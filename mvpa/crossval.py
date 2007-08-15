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

class CrossValidation( object ):
    """ Generic N-M cross-validation with arbitrary classification
    algorithms.
    """
    def __init__( self,
                  pattern,
                  trainingsamples = None,
                  testsamples = None,
                  ncvfoldsamples = 1,
                  clfcallback = None,
                  permutate = False ):
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
          permutate:  If True the regressor vector is permutated for each
                      cross-validation fold. In conjunction with ncvfoldsamples
                      this might be useful to test a classification algorithm
                      whether it can classify any noise ;)
        """
        self.__data = pattern

        # used to store the cv performances
        self.__perf = []
        self.__contingency_tbl = None

        # pattern sampling status vars
        self.__test_samplelog = []
        self.__train_samplelog = []
        self.setTrainingPatternSamples( trainingsamples )
        self.setTestPatternSamples( testsamples )
        self.setNCVFoldSamples( ncvfoldsamples )
        self.__permutated_regressors = permutate


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


    def setTrainingPatternSamples( self, samplesize ):
        """ None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__training_samplesize = samplesize


    def setTestPatternSamples( self, samplesize ):
        """ None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__test_samplesize = samplesize


    def setNCVFoldSamples( self, nsamples ):
        """ Set the number of sample runs that are performed per
        cross-validation fold.
        """
        self.__cvfold_nsamples = nsamples


    def getNCVFolds( self, cvtype ):
        """ Returns the number of cross-validations folds that are performed
        on the dataset for a given cvtype ( N-cvtype cross-validation).
        """
        return len( getUniqueLengthNCombinations(self.pattern.originlabels,
                                                 cvtype) )


    def getNPatternsPerCVFold( self, cvtype = 1 ):
        """ Returns a tuple of two arrays with the number of patterns per
        class and CV-fold. The first array lists the available training pattern
        and the second array the test patterns.

        Array rows: CV-folds, columns: regressor labels
        """
        # get the list of all combinations of to be excluded folds
        cv_list = getUniqueLengthNCombinations(self.pattern.originlabels,
                                               cvtype)

        ntrainpat = numpy.zeros( (len(cv_list), len(self.pattern.reglabels) ) )
        ntestpat = numpy.zeros( ntrainpat.shape )

        for fold, exclude in enumerate(cv_list):
            # build a boolean selector vector to choose training and
            # test data for this CV fold
            exclude_filter =  \
                numpy.array([ i in exclude for i in self.__data.origin ])

            # split data and regs into training and test set
            train = \
                self.__data.selectPatterns(
                    numpy.logical_not(exclude_filter) )
            test = self.__data.selectPatterns( exclude_filter )

            ntrainpat[fold] = train.getPatternsPerRegLabel()
            ntestpat[fold] = test.getPatternsPerRegLabel()

        return ntrainpat, ntestpat


    @staticmethod
    def splitTrainTestData( pattern, test_origin ):
        """ Split the pattern data into a training and test set.

        Parameter:
            pattern     - source MVPAPattern object
            test_origin - sequence with origin values of patterns that shall
                          be the test set.

        Returns:
            Tuple of MVPAPattern objects (train, test).
        """
        # build a boolean selector vector to choose training and
        # test data
        test_filter =  \
            numpy.array([ i in test_origin for i in pattern.origin ])

        # split data and regs into training and test set
        train = \
            pattern.selectPatterns(
                numpy.logical_not(test_filter) )
        test = pattern.selectPatterns( test_filter )

        return train, test


    @staticmethod
    def selectPatternSubset( pattern, samplesize ):
        """ Select a number of patterns for each regressor value.

        Parameter:
            pattern    - MVPAPattern object with the source patterns
            samplesize - number of to be selected patterns. Two special values
                         are recognized. None is off (all patterns are
                         selected), 'auto' sets sample size to highest
                         possible number of patterns that can be provided by
                         each class.

        Returns:
            - MVPAPattern object with the selected samples
            - Number of selected patterns per regressor class
        """
        if not samplesize == None:
            # determine number number of patterns per class
            if samplesize == 'auto':
                samplesize = \
                   numpy.array( pattern.getPatternsPerRegLabel() ).min()

            # finally select the patterns
            samples = pattern.getPatternSample( samplesize )
        else:
            # take all training patterns in the sampling run
            samples = pattern

        return samples, samplesize


    def run( self, classifier, cvtype = 1, permutate = False, **kwargs ):
        """ Perform cross-validation.

        Parameters:
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

        # get the list of all combinations of to be excluded folds
        cv_list = getUniqueLengthNCombinations(self.__data.originlabels,
                                               cvtype)
        # clean previous log data
        self.__clean_logdata()

        # do cross-validation
        for exclude in cv_list:
            # split into training and test data for this CV fold
            train, test = \
                CrossValidation.splitTrainTestData( self.__data, exclude )

            for sample in xrange( self.__cvfold_nsamples ):
                # permutate the regressors in training and test dataset
                if self.__permutated_regressors:
                    train.permutatedRegressors( True )
                    test.permutatedRegressors( True )

                # choose a training pattern sample
                train_samples, trainsamplesize = \
                    CrossValidation.selectPatternSubset( train, self.__training_samplesize )
                self.trainsamplelog.append( trainsamplesize )

                # choose a training pattern sample
                test_samples, testsamplesize = \
                    CrossValidation.selectPatternSubset( test, self.__test_samplesize )
                self.testsamplelog.append( testsamplesize )

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
                    self.makeContingencyTbl( test_samples.reg,
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


    def makeContingencyTbl(self, targets, predictions ):
        """ Create a (n x n) contingency table of two length-n vectors.

        One containing the classification targets and the other the
        corresponding predictions. The contingency table has to following
        layout:

                      predictions
                      1  2  .  .  N
                    1
                    2
          targets   .
                    .     (i,j)
                    N

        where cell (i,j) contains the absolute number of predictions j where
        the classification would have been i.
        """
        # create the contingency table template
        tbl = numpy.zeros( ( len(self.pattern.reglabels),
                             len(self.pattern.reglabels) ),
                             dtype = 'uint' )

        for t, tl in enumerate( self.pattern.reglabels ):
            for p, pl in enumerate( self.pattern.reglabels ):
                tbl[t, p] = \
                    numpy.sum( predictions[targets==tl] == pl )

        return tbl


    # read only props
    perf = property( fget=lambda self: self.__perf )
    pattern = property( fget=lambda self: self.__data )
    testsamplelog = property( fget=lambda self: self.__test_samplelog )
    trainsamplelog = property( fget=lambda self: self.__train_samplelog )
    contingencytbl = property( fget=lambda self: self.__contingency_tbl )

    # read/write props
    testsamplecfg   = property( fget=lambda self: self.__test_samplesize,
                                fset=setTestPatternSamples )
    trainsamplecfg  = property( fget=lambda self: self.__train_samplesize,
                                fset=setTrainingPatternSamples )
    ncvfoldsamples  = property( fget=lambda self: self.__cvfold_nsamples,
                                fset=setNCVFoldSamples )



def getUniqueLengthNCombinations(data, n):
    """Generates a list of lists containing all combinations of
    elements of data of length 'n' without repetitions.

        data: list
        n:    integer

    This function is adapted from a Java version posted in some forum on
    the web as an answer to the question 'How can I generate all possible
    combinations of length n?'. Unfortunately I cannot remember which
    forum it was.
    """

    # to be returned
    combos = []

    # local function that will be called recursively to collect the
    # combination elements
    def take(data, occupied, depth, taken):
        for i, d in enumerate(data):
            # only do something if this element hasn't been touch yet
            if occupied[i] == False:
                # see whether will reached the desired length
                if depth < n-1:
                    # flag the current element as touched
                    occupied[i] = True
                    # next level
                    take(data, occupied, depth+1, taken + [data[i]])
                    # 'free' the current element
                    occupied[i] == False
                else:
                    # store the final combination
                    combos.append(taken + [data[i]])
    # some kind of bitset that stores the status of each element
    # (contained in combination or not)
    occupied = [ False for i in data ]
    # get the combinations
    take(data, occupied, 0, [])

    # return the result
    return combos
