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
from mvpa2.support.copy import copy
from mvpa2.featsel.base import StaticFeatureSelection, IterativeFeatureSelection
from mvpa2.featsel.helpers import NBackHistoryStopCrit, \
                                 FixedNElementTailSelector, \
                                 BestDetector

from mvpa2.base.state import ConditionalAttribute

if __debug__:
    from mvpa2.base import debug


class IFS(IterativeFeatureSelection):
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
    def __init__(self,
                 fmeasure,
                 pmeasure,
                 splitter,
                 fselector=FixedNElementTailSelector(1, tail='upper',
                                                     mode='select'),
                 **kwargs):
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
          This splitter instance has to generate at least two dataset splits
          when called with the input dataset. The first split serves as the
          training dataset and the second as the evaluation dataset.
        """
        # bases init first
        IterativeFeatureSelection.__init__(self, fmeasure, pmeasure, splitter,
                                           fselector, **kwargs)


    def _train(self, ds):
        # local binding
        fmeasure = self._fmeasure
        fselector = self._fselector
        scriterion = self._stopping_criterion
        bestdetector = self._bestdetector

        # init
        # Computed error for each tested features set.
        errors = []
        # feature candidate are all features in the pattern object
        candidates = range(ds.nfeatures)
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

                # take the new candidate and all already selected features
                # select a new temporay feature subset from the dataset
                # slice the full dataset, because for the initial iteration
                # steps this will be much mure effecient than splitting the
                # full ds into train and test at first
                fslm = StaticFeatureSelection(selected + [candidate])
                fslm.train(ds)
                candidate_ds = fslm(ds)
                # activate the dataset splitter
                dsgen = self._splitter.generate(candidate_ds)
                # and derived the dataset part that is used for computing the selection
                # criterion
                trainds = dsgen.next()
                # compute data measure on the training part of this feature set
                measures.append(fmeasure(trainds))

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

            # actually run the performance measure to estimate "quality" of
            # selection
            fslm = StaticFeatureSelection(selected)
            fslm.train(ds)
            selectedds = fslm(ds)
            # split into train and test part
            trainds, testds = self._get_traintest_ds(selectedds)
            # evaluate and store
            error = self._evaluate_pmeasure(trainds, testds)
            errors.append(np.asscalar(error))
            # intermediate cleanup, so the datasets do not hand around while
            # the next candidate evaluation is computed
            del trainds
            del testds

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
