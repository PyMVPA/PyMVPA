### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Generic feature selection algorithms
#
#    Copyright (C) 2007 by
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
import xvalpattern

class FeatSelValidation( object ):
    """
    """
    def __init__( self, pattern, **kwargs ):
        """
        Parameters:
            pattern:

        Additional keyword arguments are passed to the CrossvalPatternGenerator
        object that is used to split the dataset. Each partial dataset is used
        to perform the feature selection (see __call__()).
        """
        self.__cvpg = \
            xvalpattern.CrossvalPatternGenerator( pattern, **(kwargs) )


    def __call__( self, featsel, classifier, **kwargs ):
        """ Runs the feature selection on each cross-validation set.

        Additional keyword arguments are passed to the cross-validation object
        for each of the inner cross-validation folds.

        Returns a list of MVPAPattern objects with the selected features from
        each cross-validation fold.
        """

        selected = []
        gen_perfs = []

        for train_samples, \
            train_samplesize, \
            test_samples, \
            test_samplesize in self.__cvpg( permutate=False ):

            select_pat,rating_map = \
                    featsel.selectFeatures( train_samples,
                                            classifier,
                                            **(kwargs) )

            selected.append( select_pat )

            classifier.train( select_pat )
            predictions = \
                classifier.predict( 
                    test_samples.selectFeaturesByMask( 
                        select_pat.getFeatureMask( copy=False) ).pattern )
            gen_perf = N.mean( predictions == test_samples.reg )
            gen_perfs.append( gen_perf )



        return selected, gen_perfs

