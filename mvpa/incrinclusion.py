### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Incremental feature inclusion algorithm
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

import numpy as np
import crossval
import sys
import stats

class IncrementalFeatureInclusion( object ):
    """ Tell me what this is!

    Two major methods: selectFeatures(), selectROIs()
    """
    def __init__( self, pattern,
                        mask,
                        cvtype = 1,
                        **kwargs ):

        """
        Parameters:
            pattern:      MVPAPattern object with the data to be analyzed.
            mask:         A mask which's size matches the origshape of the
                          patterns. This mask determines which features are
                          considered for inclusion. Depending on the used
                          algorithm the mask values have different meanings.
                          When calling selectFeatures() each nonzero mask
                          element indicates an individual selection candidate
                          feature.
                          When using selectROIs() all mask elements with a
                          common value are assumed to form a single ROI that
                          is treated as a single feature set and is included
                          at once by the algorithm.
            cvtype:       Type of cross-validation that is used. 1 means
                          N-1 cross-validation.
            **kwargs:     Additional arguments that are passed to the
                          constructor of the CrossValidation class.
        """
        self.__pattern = pattern
        self.__mask = mask
        self.__cvtype = cvtype
        self.__cvargs = kwargs

        if not mask.shape == pattern.origshape:
            raise ValueError, 'Mask shape has to match the pattern origshape.'

        self.__clearResults()


    def __clearResults( self ):
        """ Internal method used the clear a analysis results prior to the
        next run.
        """
        # init the result maps
        self.__perf = []
        self.__contingencytbl = None
        self.__selected_features = None
        self.__contingency_tbl = None
        self.__cv_perf = None


    def __doCrossValidation( self, features, clf, clfargs ):
        """ Internal method to perform a cross-validation on a subset of
        features.

        Call CrossValidation.run() and returns the CrossValidation object.
        """
        # select the feature subset from the dataset
        temp_pat = self.pattern.selectFeatures( features )

        # now do cross-validation with the current feature set
        cv = crossval.CrossValidation( temp_pat, **(self.__cvargs) )
        cv.run( clf, cvtype=self.__cvtype, **(clfargs) )

        return cv


    def __selectByFeatureId( self, selected, candidate ):
        """ Internal method: ignore! """
        return selected + [candidate]


    def __selectByMaskValue( self, selected, candidate ):
        """ Internal method: ignore! """
        return np.logical_or( selected, self.mask == candidate )


    def __doSelection( self, clf, clfargs, featureselector,
                       selected, candidates, verbose ):
        """ Backend method for selectFeatures() and selectROIs.
        """
        # yes, it is exactly that
        best_performance_ever = 0.0

        # contains the number of selected candidates
        selection_counter = 0

        # list of best canidate ids
        # in case of a simple feature selection, this is redundant with
        # the 'selected' list
        best_ids = []

        # as long as there are candidates left
        # the loop might get broken earlier if the generalization
        # error does not go down when a new ROI is selected
        while len( candidates ):
            # holds the performance value of each candidate
            candidate_rating = []

            # for all possible candidates
            for candidate in candidates:
                # display some status output about the progress if requested
                if verbose:
                    print "\rTested %i; nselected %i; mean performance: %.3f" \
                        % ( len(candidate_rating),
                            selection_counter,
                            best_performance_ever ),
                    sys.stdout.flush()

                # take the new candidate and all already selected features
                features = featureselector( selected, candidate )

                # now do cross-validation with the current feature set
                cv = self.__doCrossValidation( features,
                                               clf,
                                               clfargs )

                # store the generalization performance for this feature set
                candidate_rating.append( np.mean(cv.perf) )

            # I like arrays!
            rating_array = np.array( candidate_rating )

            # determine the best performing additonal candidate (get its id)
            best_id = candidates[ rating_array.argmax() ]

            # check if the new candidate brings no value
            # if this is the case we are done.
            if rating_array.max() < best_performance_ever:
                break

            # the new candidate adds value, because the generalization error
            # went down, therefore add it to the list of selected features
            selected = featureselector( selected, best_id )
            best_ids.append( best_id )

            # and remove it from the candidate features
            # otherwise the while loop could run forever
            candidates.remove( best_id )

            # update the latest best performance
            best_performance_ever = rating_array.max()

            # and look for the next best thing (TM)
            selection_counter += 1

        # if new candidates are left or the generalization performance
        # went down
        if verbose:
            # end pending line
            print ''

        # now do cross-validation again with the current best
        # feature set to have access to the full CV results
        cv = self.__doCrossValidation( selected,
                                       clf,
                                       clfargs )

        # and leave the loop as there is nothing more to do
        return selected, best_ids, cv



    def selectROIs( self, classifier, verbose=False, **kwargs ):
        """ Select the best set of ROIs from a mask that maximizes the
        generalization performance.

        The method works exactly like selectFeatures(), but instead of choosing
        single features from a mask it works on ROIs (sets of features).

        An ROI (region of interest) are all features sharing a common value in
        the mask.

        The method returns the list of selected ROI ids (mask values). A
        boolean mask with all selected feature from all selected ROIs is
        available via the selectionmask property.

        By setting 'verbose' to True one can enable some status messages that
        inform about the progress.

        The 'classifier' argument specifies a class that is used to perform
        the classification. Additional keyword are passed to the classifier's
        contructor.
        """
        # cleanup prior runs first
        self.__clearResults()

        # get all different mask values
        mask_values = np.unique( self.mask ).tolist()

        # selected features are stored in a mask array with the same shape
        # as the original feature mask
        selected_features = np.zeros( self.mask.shape, dtype='bool' )

        # call the backend method that returns a list of the selected
        # features and the cross-validation object with the best feature set
        # that can be used to query more detailed results
        feature_mask, selected_rois, best_cv = \
             self.__doSelection( classifier, kwargs,
                                 self.__selectByMaskValue,
                                 selected_features,
                                 mask_values,
                                 verbose )

        # store results
        self.__contingency_tbl = best_cv.contingencytbl
        self.__cv_perf = np.array( best_cv.perf )
        self.__selection_mask = feature_mask

        return selected_rois


    def selectFeatures( self, classifier, verbose=False, **kwargs ):
        """
        Select the subset of features from the feature mask that maximizes
        the classification performance on the generalization test set.

        The alorithm tests all available features in serial order with
        respect to their individiual classification performance. Only the best
        performing feature is selected. Next, each remaining feature is tested
        again whether it provides the most additional information with respect
        to classification performance. This procedure continues until a
        to-be-selected feature does not reduce the generalization error.

        By setting 'verbose' to True one can enable some status messages that
        inform about the progress.

        The 'classifier' argument specifies a class that is used to perform
        the classification. Additional keyword arguments are passed to the
        classifier's contructor.

        Returns the list of selected feature ids. Additional results are
        available via several class properties.
        """
        # cleanup prior runs first
        self.__clearResults()

        # initially empty list of selected features
        selected_features = []

        # transform coordinates of nonzero mask elements into feature ids
        cand_features = [ self.pattern.getFeatureId(coord)
                          for coord in np.transpose( self.mask.nonzero() ) ]

        # call the backend method that returns a list of the selected
        # features and the cross-validation object with the best feature set
        # that can be used to query more detailed results
        selected_features, dummy, best_cv = \
             self.__doSelection( classifier, kwargs,
                                 self.__selectByFeatureId,
                                 selected_features,
                                 cand_features,
                                 verbose )

        # store results
        self.__contingency_tbl = best_cv.contingencytbl
        self.__cv_perf = np.array( best_cv.perf )
        self.__selection_mask = \
            self.pattern.features2origmask( selected_features )

        return selected_features


    # access to the results
    selectionmask = property( fget=lambda self: self.__selection_mask )
    cvperf = property( fget=lambda self: self.__cv_perf )
    contingencytbl = property( fget=lambda self: self.__contingency_tbl )

    # other data access
    pattern = property( fget=lambda self: self.__pattern )
    mask = property( fget=lambda self: self.__mask )
    cvtype = property( fget=lambda self: self.__cvtype )

