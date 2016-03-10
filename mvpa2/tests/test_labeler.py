# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for labeler which is intended to use QE to find contigeous blobs
(AKA clusters)
"""
import numpy as np

from ..base.dataset import vstack
from ..datasets.base import Dataset
from ..measures.label import ClustersLabeler
from ..mappers.flatten import FlattenMapper
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere

from ..testing.tools import assert_equal
from ..testing.tools import assert_raises
from ..testing.tools import assert_array_equal
from ..testing.tools import assert_datasets_equal
from ..testing.tools import skip_if_no_external
from ..testing.tools import assert_array_equal_up_to_reassignment


def test_Labeler_simple():
    # Let's test first on 1d case
    ds = Dataset(np.array([[0, 1, 1, 1, 0, 0, 1, 1, 0, 1]], dtype=bool))
    # with different radius
    target_clusters  = [   [0, 1, 1, 1, 0, 0, 2, 2, 0, 3],
                           [0, 1, 1, 1, 0, 0, 2, 2, 0, 2],
                           [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]]
    ds.fa['coord'] = np.arange(ds.nfeatures)

    for radius, target in zip([1, 2, 3], target_clusters):
        labeler = ClustersLabeler(qe=IndexQueryEngine(coord=Sphere(radius)), auto_train=True)
        ds_labeled = labeler(ds)

        # TODO: most likely would do it in incremental order???
        assert_array_equal_up_to_reassignment(ds_labeled.samples, [target])

        assert_array_equal(ds_labeled.sa.maxlabels, [4-radius])

        # TODO -- check with multiple samples provided
        # weak test, but smth ;)
        ds2 = vstack([ds, ds])
        ds_labeled = labeler(ds2)
        for sample in ds_labeled.samples:
            assert_array_equal_up_to_reassignment(sample, target)

        # TODO -- check that pukes if a dataset of different nfeatures provided
        #  to call if was trained on a different ds
        labeler.untrain()
        labeler.train(ds[:, :4])
        assert_raises(ValueError, labeler, ds)

    # compare on a swarmth of random arrays of different sizes etc against
    skip_if_no_external('scipy')
    from scipy.ndimage import measurements
    # we will also do timing for internal OPT
    from time import time
    time_labeler = 0.
    time_labeler2 = 0.  # cached invocation
    time_scipy = 0.
    for ndim in range(1, 4):  # use [4] for benchmarking ;-)
        for i in range(4):
            # some random sizes, smaller for more dims
            rsize = tuple(np.random.randint(1, [0, 30, 20, 10, 80][ndim], ndim))
            # print rsize
            d = np.random.normal(size=rsize) > 0
            ds = Dataset([d])
            fm = FlattenMapper(space='coord')
            fm.train(ds)
            ds = ds.get_mapped(fm)
            assert_equal(len(ds), 1)

            # now with our labeler
            labeler = ClustersLabeler(qe=IndexQueryEngine(coord=Sphere(1)),
                                      auto_train=True,
                                      # cache=False,
                              space='our_maxlabels')
            t0 = time()
            ds_labeled = labeler(ds)
            time_labeler += time() - t0

            # the same 2nd run to take advantage of cached QE
            t0 = time()
            ds_labeled2 = labeler(ds)
            time_labeler2 += time() - t0

            # and scipy
            t0 = time()
            labels, num = measurements.label(d)
            time_scipy += time() - t0

            assert_equal(num, ds_labeled.sa.our_maxlabels[0])
            # They need to be equal but only up to reassignment of indices
            assert_array_equal_up_to_reassignment(fm.reverse(ds_labeled).samples, [labels])
            assert_datasets_equal(ds_labeled, ds_labeled2)

    # print "Timings: labeler=%.2f labeler(2nd)=%.2f scipy=%.2f" \
    #      % (time_labeler, time_labeler2, time_scipy)