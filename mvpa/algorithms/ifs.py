#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Incremental feature search (IFS).

Very similar to Recursive feature elimination (RFE), but instead of begining
with all features and stripping some sequentially, start with an empty feature
set and include important features successively.
"""

__docformat__ = 'restructuredtext'

from copy import copy

from mvpa.misc.state import State
from mvpa.algorithms.featsel import FeatureSelection, \
                                    StopNBackHistoryCriterion, \
                                    FixedNElementTailSelector

if __debug__:
    from mvpa.misc import debug


class IFS(FeatureSelection):
    """ Incremental feature search.

    A `DataMeasure` is computed multiple times on variations of a certain
    dataset. These measures are in turn used to incrementally select important
    features. Starting with an empty feature set the data measure is first
    computed for each single feature. A number of features is selected based on
    the resulting data measure map (using an `ElementSelector`).

    Next the data measure is computed again using each feature in addition to
    the already selected feature set. Again the `ElementSelector` is used to
    select more features.

    For each feature selection the transfer error on some testdatset is
    computed. This procedure is repeated until a given `StoppingCriterion`
    is reached.
    """

    def __init__(self,
                 data_measure,
                 transfer_error,
                 feature_selector=FixedNElementTailSelector(1,
                                                            tail='upper',
                                                            mode='select'),
                 stopping_criterion=StopNBackHistoryCriterion()
                 ):
        """ Initialize incremental feature search

        :Parameter:
            `data_measure`: `DataMeasure`
                Computed for each candidate feature selection.
            `transfer_error`: `TransferError`
                Compute against a test dataset for each incremental feature
                set.
            `stopping_criterion`: Functor.
                Given a list of error values it has to return a tuple of two
                booleans. First values must indicate whether the criterion is
                fulfilled and the second value signals whether the latest error
                values is the total minimum.
        """
        # bases init first
        FeatureSelection.__init__(self)

        self.__data_measure = data_measure
        self.__transfer_error = transfer_error
        self.__feature_selector = feature_selector
        self.__stopping_criterion = stopping_criterion

        # register the state members
        self._registerState("errors")


    def __call__(self, dataset, testdataset, callables=[]):
        """Proceed and select the features recursively eliminating less
        important ones.

        :Parameters:
            `dataset`: `Dataset`
                used to select features and train classifiers to determine the
                transfer error.
            `testdataset`: `Dataset`
                used to test the trained classifer on a certain feature set
                to determine the transfer error.

        Returns a new dataset with the feature subset of `dataset` that had the
        lowest transfer error of all tested sets until the stopping criterion
        was reached.
        """
        errors = []
        """Computed error for each tested features set."""

        # feature candidate are all features in the pattern object
        candidates = range( dataset.nfeatures )

        # initially empty list of selected features
        selected = []

        # results in here please
        result = None

        # as long as there are candidates left
        # the loop will most likely get broken earlier if the stopping
        # criterion is reached
        while len( candidates ):
            # measures for all candidates
            measures = []

            # for all possible candidates
            for i,candidate in enumerate(candidates):
                if __debug__:
                    debug('IFSC', "Tested %i" % i, cr=True)

                # take the new candidate and all already selected features
                # select a new temporay feature subset from the dataset
                # XXX assume MappedDataset and issue plain=True ??
                tmp_dataset = \
                    dataset.selectFeatures(selected + [candidate])

                # compute data measure on this feature set
                measures.append(self.__data_measure(tmp_dataset))

            # Select promissing feature candidates (staging)
            # IDs are only applicable to the current set of feature candidates
            tmp_staging_ids = self.__feature_selector(measures)

            # translate into real candidate ids
            staging_ids = [ candidates[i] for i in tmp_staging_ids ]

            # mark them as selected and remove from candidates
            selected += staging_ids
            for i in staging_ids:
                candidates.remove(i)

            # compute transfer error for the new set
            # XXX assume MappedDataset and issue plain=True ??
            error = self.__transfer_error(testdataset.selectFeatures(selected),
                                          dataset.selectFeatures(selected))
            errors.append(error)

            # Check if it is time to stop and if we got
            # the best result
            (stop, isthebest) = self.__stopping_criterion(errors)

            if __debug__:
                debug('IFSC',
                      "nselected %i; error: %.4f " \
                      "best/stop=%d/%d\n" \
                      % (len(selected), errors[-1], isthebest, stop),
                      cr=True, lf=True)

            if isthebest:
                # do copy to survive later selections
                results = copy(selected)

            # leave the loop when the criterion is reached
            if stop:
                break

        # charge state
        self['errors'] = errors

        # best dataset ever is returned
        return dataset.selectFeatures(results)








































##emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
##ex: set sts=4 ts=4 sw=4 et:
#### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
##
##   See COPYING file distributed along with the PyMVPA package for the
##   copyright and license terms.
##
#### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#"""Incremental feature search algorithm"""
#
#import numpy as N
#import crossval
#import sys
#
#class IFS( object ):
#    """ Incremetal search for best performing features by cross-validating each
#    of them them on the dataset.
#
#    The alorithm tests all available features in serial order with
#    respect to their individiual classification performance. Only the best
#    performing feature is selected. Next, each remaining feature is tested
#    again whether it provides the most additional information with respect
#    to classification performance. This procedure continues until a
#    to-be-selected feature does not reduce the generalization error.
#    """
#    def __init__( self,
#                  break_crit = 0.001,
#                  ntoselect=1, 
#                  verbose = False ):
#
#        """
#        Parameters:
#            break_crit:   Stop searching if the next to-be-included feature
#                          provides less than this amount of increased
#                          classification performance.
#            ntoselect:    Number of feature to be selected during each
#                          iteration.
#            verbose:      If True some status messages are enabled.
#        """
#        self.__break_crit = break_crit
#        self.__ntoselect = ntoselect
#        self.__verbose = verbose
#
#
#    def setNtoSelect( self, value ):
#        """ Set the number of feature to be selected during each iteration.
#        """
#        self.__ntoselect = value
#
#
#    def setBreakCrit( self, value ):
#        """ Stop searching if the next to-be-included feature provides less
#        than this amount of increased classification performance.
#        """
#        self.__break_crit = value
#
#
#    def selectFeatures( self, pattern, classifier, **kwargs ):
#        """ Select the subset of features from the feature mask that maximizes
#        the classification performance on the generalization test set.
#
#        Parameters:
#            pattern:      MVPAPattern object with the data to be analyzed.
#            classifier:   Classifier instance used to perform the
#                          classification.
#            ...           Additional keyword arguments are passed to the
#                          CrossValidation class.
#
#        Returns a MVPAPattern object with the selected features and a map of
#        all features (in patterns origspace; range [0,1]) with higher values
#        indication larger contributions to the classification performance.
#        """
#        # feature candidate are all features in the pattern object
#        candidates = range( pattern.nfeatures )
#
#        # yes, it is exactly that
#        best_performance_ever = 0.0
#
#        # initially empty list of selected features
#        selected = []
#
#        # assign each feature in pattern a value between 0 and 1 that reflects
#        # its contribution to the classification
#        rating_map = N.zeros( pattern.mapper.dsshape, dtype='float32' )
#
#        # selection iteration counter
#        sel_counter = 0
#
#        # as long as there are candidates left
#        # the loop might get broken earlier if the generalization
#        # error does not go down when a new ROI is selected
#        while len( candidates ):
#            # do something complicated to be able to map each candidates
#            # performance back into a map in pattern orig space
#            candidate_mask = pattern.mapper.buildMaskFromFeatureIds( candidates )
#            candidate_rating_orig = \
#                N.zeros(candidate_mask.shape, dtype='float32')
#            candidate_rating = \
#                candidate_rating_orig[ candidate_mask > 0 ]
#
#            # for all possible candidates
#            for i,candidate in enumerate(candidates):
#                # display some status output about the progress if requested
#                if self.__verbose:
#                    print "\rTested %i; nselected %i; mean performance: %.3f" \
#                        % ( i,
#                            len(selected),
#                            best_performance_ever ),
#                    sys.stdout.flush()
#
#                # take the new candidate and all already selected features
#                # select a new temporay feature subset from the dataset
#                temp_pat = \
#                    pattern.selectFeatures( selected + [candidate] )
#
#                # now do cross-validation with the current feature set
#                cv = crossval.CrossValidation( temp_pat, classifier,
#                                               **(kwargs) )
#                # run it
#                cv()
#
#                # store the generalization performance for this feature set
#                candidate_rating[i] = N.mean(cv.perf)
#
#            # check if the new candidate brings value
#            # if this is not the case we are done.
#            if candidate_rating.max() - best_performance_ever \
#               < self.__break_crit:
#                break
#
#            # determine the best performing additonal candidates (get their id)
#            best_ids = \
#                [ candidates[i] \
#                    for i in candidate_rating.argsort()[-1*self.__ntoselect:] ]
#
#            # the new candidate adds value, because the generalization error
#            # went down, therefore add it to the list of selected features
#            selected += best_ids
#
#            # and remove it from the candidate features
#            # otherwise the while loop could run forever
#            for i in best_ids:
#                candidates.remove( i )
#
#            # update the latest best performance
#            best_performance_ever = candidate_rating.max()
#            # and look for the next best thing (TM)
#
#            # map current candidate set to orig space
#            candidate_rating_orig[ candidate_mask > 0 ] = candidate_rating
#            candidate_rating_orig[ \
#                pattern.mapper.buildMaskFromFeatureIds( selected ).nonzero() ] \
#                    = best_performance_ever
#
#            # add the ratings of this iteration to the map
#            rating_map += candidate_rating_orig
#
#            # next iteration
#            sel_counter += 1
#
#        # if no candidates are left or the generalization performance
#        # went down
#        if self.__verbose:
#            # end pending line
#            print ''
#
#        # make rating_map range independent of iterations
#        rating_map /= sel_counter
#
#        # apply the final feature selection to the pattern dataset
#        return pattern.selectFeatures( selected ), rating_map
#
#
#
#    ntoselect = property( fget=lambda self: self.__ntoselect,
#                          fset=setNtoSelect )
#    breakcrit = property( fget=lambda self: self.__break_crit,
#                          fset=setBreakCrit )
