#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Generic Cross-Validation"""

import operator
from mvpa.algorithms.datameasure import DataMeasure


class ClfCrossValidation(DataMeasure):
    """
    Assumptions:
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


    def __call__(self, dataset, callbacks=[]):
        """

        Returns a sequence of postprocessingResults.
        """
        # store the results of the splitprocessor
        results = []

        # splitter
        for split in self.__splitter( dataset ):
            self.__classifier.train( split[0] )

            for splitprocessor in self.__splitprocessor:
                results.append( splitprocessor( self.__splitter,
                                                split,
                                                self.__classifier )  )

        return results
