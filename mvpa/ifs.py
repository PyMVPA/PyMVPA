### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Incremental feature search algorithm
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

        Returns a MVPAPattern object with the selected features.
        """
        # feature candidate are all features in the pattern object
        candidates = range( pattern.nfeatures )

        # yes, it is exactly that
        best_performance_ever = 0.0

        # initially empty list of selected features
        selected = []

        # as long as there are candidates left
        # the loop might get broken earlier if the generalization
        # error does not go down when a new ROI is selected
        while len( candidates ):
            # holds the performance value of each candidate
            candidate_rating = []

            # for all possible candidates
            for candidate in candidates:
                # display some status output about the progress if requested
                if self.__verbose:
                    print "\rTested %i; nselected %i; mean performance: %.3f" \
                        % ( len(candidate_rating),
                            len(selected),
                            best_performance_ever ),
                    sys.stdout.flush()

                # take the new candidate and all already selected features
                # select a new temporay feature subset from the dataset
                temp_pat = \
                    pattern.selectFeaturesById( selected + [candidate],
                                                maintain_mask = False )

                # now do cross-validation with the current feature set
                cv = crossval.CrossValidation( temp_pat, classifier,
                                               **(kwargs) )
                # run it
                cv()

                # store the generalization performance for this feature set
                candidate_rating.append( N.mean(cv.perf) )

            # I like arrays!
            rating_array = N.array( candidate_rating )

            # check if the new candidate brings value
            # if this is not the case we are done.
            if rating_array.max() - best_performance_ever < self.__break_crit:
                break

            # determine the best performing additonal candidates (get their id)
            best_ids = \
                [ candidates[i] \
                    for i in rating_array.argsort()[-1*self.__ntoselect:] ]

            # the new candidate adds value, because the generalization error
            # went down, therefore add it to the list of selected features
            selected += best_ids

            # and remove it from the candidate features
            # otherwise the while loop could run forever
            for i in best_ids:
                candidates.remove( i )

            # update the latest best performance
            best_performance_ever = rating_array.max()
            # and look for the next best thing (TM)

        # if no candidates are left or the generalization performance
        # went down
        if self.__verbose:
            # end pending line
            print ''

        # apply the final feature selection to the pattern dataset
        return pattern.selectFeatures( selected )



    ntoselect = property( fget=lambda self: self.__ntoselect,
                          fset=setNtoSelect )
    breakcrit = property( fget=lambda self: self.__break_crit,
                          fset=setBreakCrit )
