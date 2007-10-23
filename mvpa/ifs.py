### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Incremental feature search algorithm
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
import crossval
import sys

class IFS( object ):
    """ Incremetal search for best performing features by cross-validating each
    of them them on the dataset.

    The alorithm tests all available features in serial order with
    respect to their individiual classification performance. Only the best
    performing feature is selected. Next, each remaining feature is tested
    again whether it provides the most additional information with respect
    to classification performance. This procedure continues until a
    to-be-selected feature does not reduce the generalization error.
    """
    def __init__( self,
                  break_crit = 0.001,
                  ntoselect=1, 
                  verbose = False ):

        """
        Parameters:
            break_crit:   Stop searching if the next to-be-included feature
                          provides less than this amount of increased
                          classification performance.
            ntoselect:    Number of feature to be selected during each
                          iteration.
            verbose:      If True some status messages are enabled.
        """
        self.__break_crit = break_crit
        self.__ntoselect = ntoselect
        self.__verbose = verbose


    def setNtoSelect( self, value ):
        """ Set the number of feature to be selected during each iteration.
        """
        self.__ntoselect = value


    def setBreakCrit( self, value ):
        """ Stop searching if the next to-be-included feature provides less
        than this amount of increased classification performance.
        """
        self.__break_crit = value


    def selectFeatures( self, pattern, classifier, **kwargs ):
        """ Select the subset of features from the feature mask that maximizes
        the classification performance on the generalization test set.

        Parameters:
            pattern:      MVPAPattern object with the data to be analyzed.
            classifier:   Classifier instance used to perform the
                          classification.
            ...           Additional keyword arguments are passed to the
                          CrossValidation class.

        Returns a MVPAPattern object with the selected features and a map of
        all features (in patterns origspace; range [0,1]) with higher values
        indication larger contributions to the classification performance.
        """
        # feature candidate are all features in the pattern object
        candidates = range( pattern.nfeatures )

        # yes, it is exactly that
        best_performance_ever = 0.0

        # initially empty list of selected features
        selected = []

        # assign each feature in pattern a value between 0 and 1 that reflects
        # its contribution to the classification
        rating_map = N.zeros( pattern.mapper.dsshape, dtype='float32' )

        # selection iteration counter
        sel_counter = 0

        # as long as there are candidates left
        # the loop might get broken earlier if the generalization
        # error does not go down when a new ROI is selected
        while len( candidates ):
            # do something complicated to be able to map each candidates
            # performance back into a map in pattern orig space
            candidate_mask = pattern.mapper.buildMaskFromFeatureIds( candidates )
            candidate_rating_orig = \
                N.zeros(candidate_mask.shape, dtype='float32')
            candidate_rating = \
                candidate_rating_orig[ candidate_mask > 0 ]

            # for all possible candidates
            for i,candidate in enumerate(candidates):
                # display some status output about the progress if requested
                if self.__verbose:
                    print "\rTested %i; nselected %i; mean performance: %.3f" \
                        % ( i,
                            len(selected),
                            best_performance_ever ),
                    sys.stdout.flush()

                # take the new candidate and all already selected features
                # select a new temporay feature subset from the dataset
                temp_pat = \
                    pattern.selectFeatures( selected + [candidate] )

                # now do cross-validation with the current feature set
                cv = crossval.CrossValidation( temp_pat, classifier,
                                               **(kwargs) )
                # run it
                cv()

                # store the generalization performance for this feature set
                candidate_rating[i] = N.mean(cv.perf)

            # check if the new candidate brings value
            # if this is not the case we are done.
            if candidate_rating.max() - best_performance_ever \
               < self.__break_crit:
                break

            # determine the best performing additonal candidates (get their id)
            best_ids = \
                [ candidates[i] \
                    for i in candidate_rating.argsort()[-1*self.__ntoselect:] ]

            # the new candidate adds value, because the generalization error
            # went down, therefore add it to the list of selected features
            selected += best_ids

            # and remove it from the candidate features
            # otherwise the while loop could run forever
            for i in best_ids:
                candidates.remove( i )

            # update the latest best performance
            best_performance_ever = candidate_rating.max()
            # and look for the next best thing (TM)

            # map current candidate set to orig space
            candidate_rating_orig[ candidate_mask > 0 ] = candidate_rating
            candidate_rating_orig[ \
                pattern.mapper.buildMaskFromFeatureIds( selected ).nonzero() ] \
                    = best_performance_ever

            # add the ratings of this iteration to the map
            rating_map += candidate_rating_orig

            # next iteration
            sel_counter += 1

        # if no candidates are left or the generalization performance
        # went down
        if self.__verbose:
            # end pending line
            print ''

        # make rating_map range independent of iterations
        rating_map /= sel_counter

        # apply the final feature selection to the pattern dataset
        return pattern.selectFeatures( selected ), rating_map



    ntoselect = property( fget=lambda self: self.__ntoselect,
                          fset=setNtoSelect )
    breakcrit = property( fget=lambda self: self.__break_crit,
                          fset=setBreakCrit )
