### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Serial feature inclusion algorithm
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

class SerialFeatureInclusion( object ):
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


    def selectROIs( self, classifier, verbose=False, **kwargs ):
        """
        NOT IMPLEMENTED YET!
        By setting 'verbose' to True one can enable some status messages that
        inform about the progress.

        The 'classifier' argument specifies a class that is used to perform
        the classification. Additional keyword are passed to the classifier's
        contructor.
        """
        raise RuntimeError, 'Not implemented yet.'

        # cleanup prior runs first
        self.__clearResults()

        # get all different mask values
        mask_values = np.unique( self.mask )


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

        # yes, it is exactly that
        best_performance_ever = 0.0

        # as long as there are candidate features left
        # the loop might get broken earlier if the generalization
        # error does not go down when a new feature is selected
        while len( cand_features ):
            # holds the performance value of each candidate feature
            feature_rating = []

            # for all possible candidate features
            for cand_feature in cand_features:
                # display some status output about the progress if requested
                if verbose:
                    print "\rTested %i; nselected %i; mean performance: %.3f" \
                        % ( len(feature_rating),
                            len(selected_features),
                            best_performance_ever ),
                    sys.stdout.flush()

                # take the new candidate and all already selected features
                features = selected_features + [cand_feature]
                # to select the feature from the dataset
                temp_pat = self.pattern.selectFeatures( features )

                # now do cross-validation with the current feature set
                cv = crossval.CrossValidation( temp_pat, **(self.__cvargs) )
                cv.run( classifier, cvtype=self.__cvtype, **(kwargs) )

                # store the generalization performance for this feature
                feature_rating.append( np.mean(cv.perf) )

            # I like arrays!
            rating_array = np.array( feature_rating )

            # determine the best performing additonal feature (get its id)
            best_id = cand_features[ rating_array.argmax() ]

            # check if the new feature brings no value
            # if this is the case we are done.
            if rating_array.max() < best_performance_ever:
                if verbose:
                    # end pending line
                    print ''

                # now do cross-validation again with the current best
                # feature set to have access to the full CV results
                temp_pat = self.pattern.selectFeatures( selected_features )
                cv = crossval.CrossValidation( temp_pat, **(self.__cvargs) )
                cv.run( classifier, cvtype=self.__cvtype, **(kwargs) )

                # store results
                self.__selected_features = selected_features
                self.__contingency_tbl = cv.contingencytbl
                self.__cv_perf = np.array( cv.perf )

                # and leave the loop as there is nothing more to do
                break;

            # the new feature adds value, because the generalization error
            # went down, therefore add it to the list of selected features
            selected_features.append( best_id )
            # and remove it from the candidate features
            # otherwise the while loop could run forever
            cand_features.remove( best_id )

            # update the latest best performance
            best_performance_ever = rating_array.max()

            # and look for the next best thing (TM)

        return selected_features


    def getSelectionMask( self ):
        """ Returns a mask of the selected feature in the original dataspace.

        If no features were selected 'None' is returned.
        """
        if self.selectedfeatures == None:
            return None

        return self.pattern.features2origmask( self.selectedfeatures )


    # access to the results
    selectedfeatures = property( fget=lambda self: self.__selected_features )
    selectionmask = property( fget=getSelectionMask )
    cvperf = property( fget=lambda self: self.__cv_perf )
    contingencytbl = property( fget=lambda self: self.__contingency_tbl )

    # other data access
    pattern = property( fget=lambda self: self.__pattern )
    mask = property( fget=lambda self: self.__mask )
    cvtype = property( fget=lambda self: self.__cvtype )

