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
    def __init__( self, pattern, classifier, **kwargs ):
        """
        Initialize the cross-validation.

          pattern:    MVPAPattern object containing the data, classification
                      targets and origins for the cross-validation.
          classifier: class that shall be used the actual classification. Its
                      constructor must not have more than two required
                      arguments (data and regs - in this order).
                      The classifier has to train itself when creating the
                      classifier object!
          **kwargs:   All keyword arguments are passed to the classifiers
                      constructor.
        """
        self.__data = pattern

        # check and store the classifier
        self.setClassifier( classifier, **(kwargs) )

        # used to store the cv performances
        self.__perf = []
        self.__contingency_tbl = None

        # pattern sampling status vars
        self.__training_samplesize = None
        self.__test_samplesize = None
        self.__cvfold_nsamples = 1
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


    def setClassifier( self, classifier, **kwargs ):
        """ The supplied values overwrite those passed to the contructor.
        """
        if not hasattr( classifier, 'predict' ):
            raise ValueError, "Classifier object has to provide a " \
                         "'predict()' method."

        self.__clf = classifier
        self.__clf_kwargs = kwargs


    def setTrainingPatternSampling( self, samplesize = 'auto' ):
        """ None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__training_samplesize = samplesize


    def setTestPatternSampling( self, samplesize = 'auto' ):
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


    def run( self, cvtype = 1):
        """ Start cross-validation function.

        Parameters:
          cvtype:         type of cross-validation: N-cv

        Returns:
          List of performance values (fraction of correct classifications) for
          each  cross-validation fold.
        """
        # get the list of all combinations of to be excluded folds
        cv_list = getUniqueLengthNCombinations(self.__data.originlabels,
                                               cvtype)
        # clean previous log data
        self.__clean_logdata()

        # do cross-validation
        for exclude in cv_list:
            # build a boolean selector vector to choose training and
            # test data for this CV fold
            exclude_filter =  \
                numpy.array([ i in exclude for i in self.__data.origin ])

            # split data and regs into training and test set
            train = \
                self.__data.selectPatterns( 
                    numpy.logical_not(exclude_filter) )
            test = self.__data.selectPatterns( exclude_filter )

            for sample in xrange( self.__cvfold_nsamples ):
                # choose a training pattern sample
                if not self.__training_samplesize == None:
                    # determine number number of patterns per class
                    if self.__training_samplesize == 'auto':
                        trainsamplesize =\
                           numpy.array( train.getPatternsPerRegLabel() ).min()
                    else:
                        # take predefined number of patterns
                        trainsamplesize = self.__training_samplesize

                    # finally select the patterns
                    train_samples = \
                        train.getPatternSample( trainsamplesize )
                    self.trainsamplelog.append( trainsamplesize )

                else:
                    # take all training patterns in the sampling run
                    train_samples = train
                    self.trainsamplelog.append( None )

                # choose a test pattern sample
                if not self.__test_samplesize == None:
                    # determine number number of patterns per class
                    if self.__test_samplesize == 'auto':
                        # choose the minimum number of patterns that is
                        # available for all classes
                        testsamplesize =\
                           numpy.array( test.getPatternsPerRegLabel() ).min()
                    else:
                        # take predefined number of patterns
                        testsamplesize = self.__test_samplesize

                    # finally select the patterns
                    test_samples = \
                        test.getPatternSample( testsamplesize )
                    self.testsamplelog.append( testsamplesize )

                else:
                    # take all test patterns in this sampling run
                    test_samples = test
                    self.testsamplelog.append( None )

                # create classifier (must include training if necessary)
                clf = self.__clf(train_samples )

                # test
                predictions = numpy.array(clf.predict(test_samples.pattern))
                perf = ( predictions == test_samples.reg )

                # store performance
                self.perf.append(perf.mean())

                # store more details performance description
                # as contingency table
                self.__storeUpdateContingencyTbl( 
                    self.makeContingencyTbl( test_samples.reg,
                                             predictions ) )

        return self.perf


    def __storeUpdateContingencyTbl( self, tbl ):
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
                      1 2 . . . N
                    1
                    2
          targets   .
                    .
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
                                fset=setTestPatternSampling )
    trainsamplecfg  = property( fget=lambda self: self.__train_samplesize,
                                fset=setTrainingPatternSampling )
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
