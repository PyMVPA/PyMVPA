### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Generic Cross-Validation
#
#    Copyright (C) 2006-2007 by
#    Michael Hanke <michael.hanke@gmail.com>
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

import operator


class CrossValidation(object):
    """
    """
    def __init__( self,
                  splitter,
                  classifier,
                  splitprocessor ):
        """
        @splitprocessor  --- list of instances which gets arguments:
                             generated split, splitter object, classifier
        """
        self.__splitter = splitter
        self.__classifier = classifier

        # make sure we always deal with sequences
        if not operator.isSequenceType( splitprocessor ):
            self.__splitprocessor = [ splitprocessor ]
        else:
            self.__splitprocessor = splitprocessor


    def __call__( self, dataset ):
        """

        Returns a sequence of postprocessingResults.
        """
        # store the results of the splitprocessor
        results = []

        # splitter
        for split in self.__splitter( dataset ):
            classifier.train( split[0] )

            for splitprocessor in self.__splitprocessor:
                results.append( splitprocessor( self.__splitter,
                                                split,
                                                classifier )  )

        return results
