# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA searchlight algorithm"""

from mvpa.base import externals
from mvpa.measures.searchlight import sphere_searchlight
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError

from tests_warehouse import *
from tests_warehouse_clfs import *

class SearchlightTests(unittest.TestCase):

    def setUp(self):
        self.dataset = datasets['3dlarge']
        # give the feature coord a more common name, matching the default of
        # the searchlight
        self.dataset.fa['voxel_indices'] = self.dataset.fa.myspace


    def testSpatialSearchlight(self):
        # compute N-1 cross-validation for each sphere
        transerror = TransferError(sample_clf_lin)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1))

        sls = [sphere_searchlight(cv, radius=1, transformer=N.array,
                         enable_states=['roisizes', 'raw_results'])]

        if externals.exists('pprocess'):
            sls += [sphere_searchlight(cv, radius=1, transformer=N.array,
                         nproc=2,
                         enable_states=['roisizes', 'raw_results'])]

        for sl in sls:
            # run searchlight
            results = sl(self.dataset)

            # check for correct number of spheres
            self.failUnless(len(results) == 106)

            # check for chance-level performance across all spheres
            self.failUnless(0.4 < results.mean() < 0.6)

            # check resonable sphere sizes
            self.failUnless(len(sl.states.roisizes) == 106)
            self.failUnless(max(sl.states.roisizes) == 7)
            self.failUnless(min(sl.states.roisizes) == 4)

            # check base-class state
            self.failUnlessEqual(len(sl.states.raw_results), 106)


    def testPartialSearchlightWithFullReport(self):
        # compute N-1 cross-validation for each sphere
        transerror = TransferError(sample_clf_lin)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                combiner=N.array)
        # contruct diameter 1 (or just radius 0) searchlight
        sl = sphere_searchlight(cv, radius=0, transformer=N.array,
                         center_ids=[3,50])

        # run searchlight
        results = sl(self.dataset)

        # only two spheres but error for all CV-folds
        self.failUnlessEqual(results.shape, (2, len(self.dataset.UC)))


    def testChiSquareSearchlight(self):
        # only do partial to save time
        if not externals.exists('scipy'):
            return

        from mvpa.misc.stats import chisquare

        transerror = TransferError(sample_clf_lin)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                enable_states=['confusion'])


        def getconfusion(data):
            cv(data)
            return chisquare(cv.states.confusion.matrix)[0]

        sl = sphere_searchlight(getconfusion, radius=0,
                         center_ids=[3,50])

        # run searchlight
        results = sl(self.dataset)

        self.failUnless(len(results) == 2)



def suite():
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':
    import runner

