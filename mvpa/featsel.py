#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Generic feature selection algorithm frontend"""

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

        # reset status vars first
        self.__selected = []
        self.__rating_maps = []
        self.__perfs = []

        # for all cross validation folds
        for train_samples, \
            train_samplesize, \
            test_samples, \
            test_samplesize in self.__cvpg( permute=False ):

            # perform feature selection and store the MVPAPattern
            # with the selected features of the CV fold and its associated
            # rating map for _all_ features in the original pattern object
            select_pat,rating_map = \
                    featsel.selectFeatures( train_samples,
                                            classifier,
                                            **(kwargs) )
            self.__selected.append( select_pat )
            self.__rating_maps.append( rating_map)

            # finally test selected features against the training set
            # and store validation performance
            classifier.train( select_pat )
            predictions = \
                classifier.predict( 
                    test_samples.selectFeaturesByMask( 
                        select_pat.mapper.getMask( copy=False) ).samples )
            self.__perfs.append( N.mean( predictions == test_samples.regs ) )


    def getMeanRatingMap( self ):
        return N.mean( self.rating_maps, axis=0 )


    def getFeatureMasks( self ):
        return [ s.getFeatureMask(copy=True) for s in self.selections ]


    xvalpattern    = property( fget=lambda self: self.__cvpg )
    selections     = property( fget=lambda self: self.__selected )
    rating_maps    = property( fget=lambda self: self.__rating_maps )
    perfs          = property( fget=lambda self: self.__perfs )
