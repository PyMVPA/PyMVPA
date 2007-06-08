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
    def __init__( self, pattern, classifier, **kwargs ):
        """
        Initialize the cross-validation.

          classifier: class that shall be used the actual classification. Its
                      constructor must not have more than two required
                      arguments (data and regs - in this order).
                      The classifier has to train itself by creating the
                      classifier object!
          **kwargs:   All keyword arguments are passed to the classifiers
                      constructor.
        """
        self.__data = pattern

        # check and store the classifier
        self.setClassifier( classifier, **(kwargs) )


    def setClassifier( self, classifier, **kwargs ):
        if not hasattr( classifier, 'predict' ):
            raise ValueError, "Classifier object has to provide a " \
                         "'predict()' method."

        self.__clf = classifier
        self.__clf_kwargs = kwargs


    def start( self, cv = 1, classifier = None, **kwargs):
        """ Start cross-validation function.

        Parameters:
          cv:         type of cross-validation: N-cv

        Returns:
          List of performance values (fraction of correct classifications) for
          each  cross-validation fold.
        """
        # change classifier if requested
        if classifier:
            self.setClassifier( classifier, **(kwargs) )

        # get the list of all combinations of to be excluded folds
        cv_list = getUniqueLengthNCombinations(self.__data.originlabels, cv)
        print cv_list

        performance = []

        # do cross-validation
        for exclude in cv_list:
            # build a boolean selector vector to choose training and test data
            # for this CV fold
            exclude_filter =  \
                numpy.array( [ i in exclude for i in self.__data.origin ] )

            # split data and regs into training and test set
            train = \
                self.__data.selectPatterns( numpy.logical_not(exclude_filter))
            test = self.__data.selectPatterns( exclude_filter )

            # create classifier (must include training if necessary)
            clf = self.__clf(train, **(kwargs) )

            # test
            perf = numpy.array(clf.predict(test.pattern))
            perf = perf == test.reg

            # store performance
            performance.append(perf.mean())

        return performance


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
