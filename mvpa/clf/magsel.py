### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Feature selcetion by weight magnitude
#
#    Copyright (C) 2007 by
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

import numpy as N
import support


class MagnitudeFeatureSelection( object ):
    """
    """
    def __init__( self,
                  ntoselect,
                  select_by = 'fraction',
                  verbose=False ):
        """
        """
        self.__ntoselect = ntoselect
        self.__select_by = select_by
        self.__verbose = verbose


    def selectFeatures( self, pattern, classifier ):
        """
        """
        # train clf on given dataset and store performance on training set
        classifier.train( pattern )

        # feature rating: large values == important
        fr = classifier.getFeatureBenchmark()

        # determine the number of features to be selected
        if self.__select_by == 'fraction':
            nselect = int( len(featrank) * self.__ntoselect )
        elif self.__select_by == 'number':
            nselect = self.__ntoselect
        else:
            raise ValueError, 'Unknown elimination method: ' \
                              + str(self.__select_by)

        if self.__verbose:
            print "Selecting", nselect, "features."

        # map the weights into the orig feature space
        weight_map = N.zeros( pattern.origshape, dtype='float32' )
        weight_map[pattern.getFeatureMask(copy=False) > 0] = fr

        # return the first 'nselect' features (with the highest ranking)
        # and the map of all feature weights
        return pattern.selectFeaturesById( fr.argsort()[:nselect] ), weight_map


    def setVerbosity( self, value ):
        """ If verbosity is set to True, some status messages will be printed
        during recursive feature selection.
        """
        self.__verbose = value


    # properties
    verbose = property( fget=lambda self: self.__verbose,
                        fset=setVerbosity )
