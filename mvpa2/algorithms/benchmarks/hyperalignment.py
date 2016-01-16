# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Benchmarks for hyperalignment algorithms
"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.base import warning
from mvpa2.support import copy
from mvpa2.mappers.base import IdentityMapper
from mvpa2.mappers.zscore import zscore
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.generators.splitters import Splitter
from mvpa2.mappers.boxcar import BoxcarMapper
from mvpa2.datasets.base import FlattenMapper
from mvpa2.measures.anova import vstack
from mvpa2.mappers.fx import mean_group_sample

if __debug__:
    from mvpa2.base import debug

def wipe_out_offdiag(a, window_size, value=np.inf):
    """Wipe-out (fill with np.inf, as default) close-to-diagonal elements

    Parameters
    ----------
    a : array
    window_size : int
      How many "off-diagonal" elements to preserve
    value : optional
      Value to fill in with
    """
    for r, _ in enumerate(a):
        a[r, max(0, r - window_size):r] = value
        a[r, r + 1:min(len(a), r + window_size)] = value
    return a


def timesegments_classification(
        dss,
        hyper=None,
        part1=HalfPartitioner(),
        part2=NFoldPartitioner(attr='subjects'),
        window_size=6,
        overlapping_windows=True,
        distance='correlation',
        do_zscore=True):
    """Time-segment classification across subjects using Hyperalignment

    Parameters
    ----------
    dss : list of datasets
       Datasets to benchmark on.  Usually a single dataset per subject.
    hyper : Hyperalignment-like, optional
       Beast which if called on a list of datasets should spit out trained
       mappers.  If not specified, `IdentityMapper`s will be used
    part1 : Partitioner, optional
       Partitioner to split data for hyperalignment "cross-validation"
    part2 : Partitioner, optional
       Partitioner for CV within the hyperalignment test split
    window_size : int, optional
       How many temporal points to consider for a classification sample
    overlapping_windows : bool, optional
       Strategy to how create and classify "samples" for classification.  If
       True -- `window_size` samples from each time point (but trailing ones)
       constitute a sample, and upon "predict" `window_size` of samples around
       each test point is not considered.  If False -- samples are just taken
       (with training and testing splits) at `window_size` step from one to
       another.
    do_zscore : bool, optional
       Perform zscoring (overall, not per-chunk) for each dataset upon
       partitioning with part1
    ...
    """
    # Generate outer-most partitioning ()
    parts = [copy.deepcopy(part1).generate(ds) for ds in dss]

    iter = 1
    errors = []

    while True:
        try:
            dss_partitioned = [p.next() for p in parts]
        except StopIteration:
            # we are done -- no more partitions
            break
        if __debug__:
            debug("BM", "Iteration %d", iter)

        dss_train, dss_test = zip(*[list(Splitter("partitions").generate(ds))
                                    for ds in dss_partitioned])

        # TODO:  allow for doing feature selection

        if do_zscore:
            for ds in dss_train + dss_test:
                zscore(ds, chunks_attr=None)

        if hyper is not None:
            # since otherwise it would remember previous loop dataset as the "commonspace"
            # Now let's do hyperalignment but on a copy in each loop iteration
            hyper_ = copy.deepcopy(hyper)
            mappers = hyper_(dss_train)
        else:
            mappers = [IdentityMapper() for ds in dss_train]

        dss_test_aligned = [mapper.forward(ds) for mapper, ds in zip(mappers, dss_test)]

        # assign .sa.subjects to those datasets
        for i, ds in enumerate(dss_test_aligned):
            # part2.attr is by default "subjects"
            ds.sa[part2.attr] = [i]

        dss_test_bc = []
        for ds in dss_test_aligned:
            if overlapping_windows:
                startpoints = range(len(ds) - window_size + 1)
            else:
                startpoints = _get_nonoverlapping_startpoints(len(ds), window_size)
            bm = BoxcarMapper(startpoints, window_size)
            bm.train(ds)
            ds_ = bm.forward(ds)
            ds_.sa['startpoints'] = startpoints
            # reassign subjects so they are not arrays
            def assign_unique(ds, sa):
                ds.sa[sa] = [np.asscalar(np.unique(x)) for x in ds.sa[sa].value]
            assign_unique(ds_, part2.attr)

            fm = FlattenMapper()
            fm.train(ds_)
            dss_test_bc.append(ds_.get_mapped(fm))

        ds_test = vstack(dss_test_bc)
        # Perform classification across subjects comparing against mean
        # spatio-temporal pattern of other subjects
        errors_across_subjects = []
        for ds_test_part in part2.generate(ds_test):
            ds_train_, ds_test_ = list(Splitter("partitions").generate(ds_test_part))
            # average across subjects to get a representative pattern per timepoint
            ds_train_ = mean_group_sample(['startpoints'])(ds_train_)
            assert(ds_train_.shape == ds_test_.shape)

            if distance == 'correlation':
                # TODO: redo more efficiently since now we are creating full
                # corrcoef matrix.  Also we might better just take a name for
                # the pdist measure but then implement them efficiently
                # (i.e. without hstacking both pieces together first)
                dist = 1 - np.corrcoef(ds_train_, ds_test_)[len(ds_test_):, :len(ds_test_)]
            else:
                raise NotImplementedError

            if overlapping_windows:
                dist = wipe_out_offdiag(dist, window_size)

            winners = np.argmin(dist, axis=1)
            error = np.mean(winners != np.arange(len(winners)))
            errors_across_subjects.append(error)
        errors.append(errors_across_subjects)
        iter += 1

    errors = np.array(errors)
    if __debug__:
        debug("BM", "Finished with %s array of errors. Mean error %.2f"
              % (errors.shape, np.mean(errors)))
    return errors


def _get_nonoverlapping_startpoints(n, window_size):
    return range(0, n - window_size + 1, window_size)
