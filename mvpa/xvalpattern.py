### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: General N-M cross-validation pattern generator
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

import numpy as np
import support

class CrossvalPatternGenerator( object ):
    """ Generic N-M cross-validation with arbitrary classification
    algorithms.
    """
    def __init__( self,
                  pattern,
                  trainingsamples = None,
                  testsamples = None,
                  ncvfoldsamples = 1,
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
          permutate:  If True the regressor vector is permutated for each
                      cross-validation fold. In conjunction with ncvfoldsamples
                      this might be useful to test a classification algorithm
                      whether it can classify any noise ;)
        """
        self.__data = pattern

        # pattern sampling status vars
        self.setTrainingPatternSamples( trainingsamples )
        self.setTestPatternSamples( testsamples )
        self.setNCVFoldSamples( ncvfoldsamples )
        self.__permutated_regressors = permutate


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

        ntrainpat = np.zeros( (len(cv_list), len(self.pattern.reglabels) ) )
        ntestpat = np.zeros( ntrainpat.shape )

        for fold, exclude in enumerate(cv_list):
            # build a boolean selector vector to choose training and
            # test data for this CV fold
            exclude_filter =  \
                np.array([ i in exclude for i in self.__data.origin ])

            # split data and regs into training and test set
            train = \
                self.__data.selectPatterns(
                    np.logical_not(exclude_filter) )
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
            np.array([ i in test_origin for i in pattern.origin ])

        # split data and regs into training and test set
        train = \
            pattern.selectPatterns(
                np.logical_not(test_filter) )
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
                   np.array( pattern.getPatternsPerRegLabel() ).min()

            # finally select the patterns
            samples = pattern.getPatternSample( samplesize )
        else:
            # take all training patterns in the sampling run
            samples = pattern

        return samples, samplesize


    def __call__( self, cvtype = 1, permutate = False ):
        """ Perform cross-validation.

        Parameter:
          cvtype:     Type of cross-validation: N-cvtype
        """
        # get the list of all combinations of to be excluded folds
        cv_list = \
            support.getUniqueLengthNCombinations(self.__data.originlabels,
                                                 cvtype)

        # do cross-validation
        for exclude in cv_list:
            # split into training and test data for this CV fold
            train, test = \
                CrossvalPatternGenerator.splitTrainTestData( self.__data, exclude )

            # do the sampling for this CV fold
            for sample in xrange( self.__cvfold_nsamples ):
                # permutate the regressors in training and test dataset
                if self.__permutated_regressors:
                    train.permutatedRegressors( True )
                    test.permutatedRegressors( True )

                # choose a training pattern sample
                train_samples, train_samplesize = \
                    CrossvalPatternGenerator.selectPatternSubset( train,
                                                    self.__training_samplesize )

                # choose a test pattern sample
                test_samples, testsamplesize = \
                    CrossvalPatternGenerator.selectPatternSubset( test, self.__test_samplesize )

                yield train_samples, test_samples


    # read only props
    pattern = property( fget=lambda self: self.__data )

    # read/write props
    testsamplesize  = property( fget=lambda self: self.__test_samplesize,
                                fset=setTestPatternSamples )
    trainsamplesize = property( fget=lambda self: self.__train_samplesize,
                                fset=setTrainingPatternSamples )
    ncvfoldsamples  = property( fget=lambda self: self.__cvfold_nsamples,
                                fset=setNCVFoldSamples )



