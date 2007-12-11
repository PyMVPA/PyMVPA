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

from mvpa.algorithms.featsel import FeatureSelection, \
                                    StopNBackHistoryCriterion, \
                                    FixedNElementTailSelector

if __debug__:
    from mvpa.misc import debug


class IFS(FeatureSelection):
    """Incremental feature search.

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
        """Initialize incremental feature search

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

        Returns a tuple with the dataset containing the feature subset of
        `dataset` that had the lowest transfer error of all tested sets until
        the stopping criterion was reached. The tuple also contains a dataset
        with the corrsponding features from the `testdataset`.
        """
        errors = []
        """Computed error for each tested features set."""

        # feature candidate are all features in the pattern object
        candidates = range( dataset.nfeatures )

        # initially empty list of selected features
        selected = []

        # results in here please
        results = None

        # as long as there are candidates left
        # the loop will most likely get broken earlier if the stopping
        # criterion is reached
        while len( candidates ):
            # measures for all candidates
            measures = []

            # for all possible candidates
            for i, candidate in enumerate(candidates):
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
        return dataset.selectFeatures(results), \
               testdataset.selectFeatures(results)
