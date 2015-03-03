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

from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.generators.splitters import Splitter

def zero_out_offdiag(dist, window_size):
    for r in range(len(dist)):
        dist[r, max(0, r-window_size):r] = np.inf
        dist[r, r+1:min(len(dist), r+window_size)] = np.inf
    return dist

def _zero_out_offdiag(dist, window_size):
    for i in xrange(len(dist)):
        for j in xrange(len(dist)):
            if abs(i-j) < window_size and i!=j:
                dist[i, j] = np.inf
    return dist

def timesegments_classification(
        dss,
        hyper,
        part1=HalfPartitioner(),  # partitioner to split data for hyperalignment
        part2=NFoldPartitioner(attr='subjects'), # partitioner for CV in the test split
        window_size=6,
        overlapping_windows=True,
        distance='correlation',
        do_zscore=False,
        clf_direction_correct_way=True):
    """Time-segment classification across subjects using Hyperalignment
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
        print "iteration ", iter
        dss_train, dss_test = zip(*[list(Splitter("partitions").generate(ds))
                                    for ds in dss_partitioned])

        # TODO:  allow for doing feature selection
        # Now let's do hyperalignment
        hyper = copy.deepcopy(hyper)
        if do_zscore:
            for ds in dss_train + dss_test:
                zscore(ds, chunks_attr=None)

        mappers = hyper(dss_train)

        # dss_train_aligned = [mapper.forward(ds) for mapper, ds in zip(mappers, dss_train)]
        dss_test_aligned = [mapper.forward(ds) for mapper, ds in zip(mappers, dss_test)]

        # assign .sa.subjects to those datasets
        for i, ds in enumerate(dss_test_aligned):  ds.sa["subjects"] = [i]

        dss_test_bc = []
        for ds in dss_test_aligned:
            if overlapping_windows:
                startpoints = range(len(ds) - window_size)
            else:
                raise NotImplementedError
            ds_ = BoxcarMapper(startpoints, window_size).forward(ds)
            ds_.sa['startpoints'] = startpoints
            # reassign subjects so they are not arrays
            def assign_unique(ds, sa):
                ds.sa[sa] = [np.asscalar(np.unique(x)) for x in ds.sa[sa].value]
            for saname in ['subjects']:
                assign_unique(ds_, saname)

            fm = FlattenMapper(); fm.train(ds_)
            dss_test_bc.append(ds_.get_mapped(fm))

        ds_test = vstack(dss_test_bc)
        # Perform classification across subjects comparing against mean
        # spatio-temporal pattern of other subjects
        errors_across_subjects = []
        for ds_test_part in part2.generate(ds_test):
            ds_train_, ds_test_ = list(Splitter("partitions").generate(ds_test_part))
            ds_train_ = mean_group_sample(['startpoints'])(ds_train_)
            assert(ds_train_.shape == ds_test_.shape)

            if distance == 'correlation':
                # TODO: redo
                dist = 1 - np.corrcoef(ds_train_, ds_test_)[len(ds_test_):, :len(ds_test_)]
            else:
                raise NotImplementedError

            if overlapping_windows:
                dist = zero_out_offdiag(dist, window_size)

            # For Yarik it feels that we need axis=1 here!! TODO
            winners = np.argmin(dist, axis=int(clf_direction_correct_way))
            #print winners
            error = np.mean(winners != np.arange(len(winners)))
            #print error
            errors_across_subjects.append(error)
        errors.append(errors_across_subjects)
        iter += 1
    errors = np.array(errors)
    if __debug__:
        debug("BM", "Finished with %s array of errors. Mean error %.2f"
              % (errors.shape, np.mean(errors)))
    return errors