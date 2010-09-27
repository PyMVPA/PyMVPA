# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

import numpy as np
from mvpa.support.copy import copy

from mvpa.featsel.base import FeatureSelection
from mvpa.featsel.helpers import NBackHistoryStopCrit, \
                                 FixedNElementTailSelector, \
                                 BestDetector

from mvpa.base.state import ConditionalAttribute

if __debug__:
    from mvpa.base import debug


class IFS(FeatureSelection):
    """Incremental feature search.

    A scalar `Measure` is computed multiple times on variations of a
    certain dataset. These measures are in turn used to incrementally select
    important features. Starting with an empty feature set the dataset measure
    is first computed for each single feature. A number of features is selected
    based on the resulting data measure map (using an `ElementSelector`).

    Next the dataset measure is computed again using each feature in addition
    to the already selected feature set. Again the `ElementSelector` is used to
    select more features.

    For each feature selection the transfer error on some testdatset is
    computed. This procedure is repeated until a given `StoppingCriterion`
    is reached.
    """

    errors = ConditionalAttribute()

    def __init__(self,
                 feature_measure,
                 performance_measure,
                 splitter,
                 bestdetector=BestDetector(),
                 stopping_criterion=NBackHistoryStopCrit(BestDetector()),
                 feature_selector=FixedNElementTailSelector(1,
                                                            tail='upper',
                                                            mode='select'),
                 **kwargs
                 ):
        """Initialize incremental feature search

        Parameters
        ----------
        feature_measure : Measure
          Computed for each candidate feature selection. The measure has
          to compute a scalar value.
        performance_measure : Measure
          Compute against a test dataset for each incremental feature
          set.
        splitter: Splitter
          This splitter instance has to generate at least one dataset split
          when called with the input dataset that is used to compute the
          per-feature criterion for feature selection.
        bestdetector : Functor
          Given a list of error values it has to return a boolean that
          signals whether the latest error value is the total minimum.
        stopping_criterion : Functor
          Given a list of error values it has to return whether the
          criterion is fulfilled.
        feature_selector
        """
        # bases init first
        FeatureSelection.__init__(self, **kwargs)

        self.__feature_measure = feature_measure
        self.__performance_measure = performance_measure
        self.__splitter = splitter
        self.__feature_selector = feature_selector
        self.__bestdetector = bestdetector
        self.__stopping_criterion = stopping_criterion


    def _train(self, ds):
        # local binding
        splitter = self.__splitter
        pmeasure = self.__performance_measure
        fmeasure = self.__feature_measure
        fselector = self.__feature_selector
        scriterion = self.__stopping_criterion
        bestdetector = self.__bestdetector

        # activate the dataset splitter
        dsgen = splitter.generate(ds)
        # and derived the dataset part that is used for computing the selection
        # criterion
        dataset = dsgen.next()

        errors = []
        """Computed error for each tested features set."""

        # feature candidate are all features in the pattern object
        candidates = range(dataset.nfeatures)

        # initially empty list of selected features
        selected = []

        # results in here please
        results = None

        # as long as there are candidates left
        # the loop will most likely get broken earlier if the stopping
        # criterion is reached
        while len(candidates):
            # measures for all candidates
            measures = []

            # for all possible candidates
            for i, candidate in enumerate(candidates):
                if __debug__:
                    debug('IFSC', "Tested %i" % i, cr=True)

                # XXX perform feature selection with a dedicated FeatureSliceMapper

                # take the new candidate and all already selected features
                # select a new temporay feature subset from the dataset
                # XXX assume MappedDataset and issue plain=True ??
                tmp_dataset = \
                        dataset[:, selected + [candidate]]

                # compute data measure on this feature set
                measures.append(fmeasure(tmp_dataset))

            # relies on ds.item() to work properly
            measures = [np.asscalar(m) for m in measures]

            # Select promissing feature candidates (staging)
            # IDs are only applicable to the current set of feature candidates
            tmp_staging_ids = fselector(measures)

            # translate into real candidate ids
            staging_ids = [candidates[i] for i in tmp_staging_ids]

            # mark them as selected and remove from candidates
            selected += staging_ids
            for i in staging_ids:
                candidates.remove(i)

            # compute transfer error for the new set
            # XXX perform feature selection with a dedicated FeatureSliceMapper
            # XXX assume MappedDataset and issue plain=True ??
            error = pmeasure(ds[:, selected])
            errors.append(np.asscalar(error))

            # Check if it is time to stop and if we got
            # the best result
            stop = scriterion(errors)
            isthebest = bestdetector(errors)

            if __debug__:
                debug('IFSC',
                      "nselected %i; error: %.4f " \
                      "best/stop=%d/%d\n" \
                      % (len(selected), errors[-1], isthebest, stop),
                      cr=True, lf=True)

            if isthebest:
                # announce desired features to the underlying slice mapper
                # do copy to survive later selections
                self._safe_assign_slicearg(copy(selected))

            # leave the loop when the criterion is reached
            if stop:
                break

        # charge state
        self.ca.errors = errors
