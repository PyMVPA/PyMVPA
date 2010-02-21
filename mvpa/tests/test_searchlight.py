# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA searchlight algorithm"""

from mvpa.testing import *
from mvpa.testing.clfs import *
from mvpa.testing.datasets import *

from mvpa.base import externals
from mvpa.measures.searchlight import sphere_searchlight
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError
from mvpa.clfs.gnb import GNB


class SearchlightTests(unittest.TestCase):

    def setUp(self):
        self.dataset = datasets['3dlarge']
        # give the feature coord a more common name, matching the default of
        # the searchlight
        self.dataset.fa['voxel_indices'] = self.dataset.fa.myspace


    def test_spatial_searchlight(self):
        # compute N-1 cross-validation for each sphere
        # YOH: unfortunately sample_clf_lin is not guaranteed
        #      to provide exactly the same results due to inherent
        #      iterative process.  Therefore lets use something quick
        #      and pure Python
        transerror = TransferError(GNB(common_variance=True))
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1))

        sls = [sphere_searchlight(cv, radius=1,
                         enable_ca=['roisizes', 'raw_results'])]

        if externals.exists('pprocess'):
            sls += [sphere_searchlight(cv, radius=1,
                         nproc=2,
                         enable_ca=['roisizes', 'raw_results'])]

        all_results = []
        for sl in sls:
            # run searchlight
            results = sl(self.dataset)
            all_results.append(results)

            # check for correct number of spheres
            self.failUnless(results.nfeatures == 106)
            # and measures (one per xfold)
            self.failUnless(len(results) == len(self.dataset.UC))

            # check for chance-level performance across all spheres
            self.failUnless(0.4 < results.samples.mean() < 0.6)

            # check resonable sphere sizes
            self.failUnless(len(sl.ca.roisizes) == 106)
            self.failUnless(max(sl.ca.roisizes) == 7)
            self.failUnless(min(sl.ca.roisizes) == 4)

            # check base-class state
            self.failUnlessEqual(sl.ca.raw_results.nfeatures, 106)

        if len(all_results) > 1:
            # if we had multiple searchlights, we can check either they all
            # gave the same result (they should have)
            aresults = N.array([a.samples for a in all_results])
            dresults = N.abs(aresults - aresults.mean(axis=0))
            dmax = N.max(dresults)
            self.failUnlessEqual(dmax, 0.0)

    def test_partial_searchlight_with_full_report(self):
        # compute N-1 cross-validation for each sphere
        transerror = TransferError(sample_clf_lin)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1))
        # contruct diameter 1 (or just radius 0) searchlight
        sl = sphere_searchlight(cv, radius=0,
                         center_ids=[3,50])

        # run searchlight
        results = sl(self.dataset)

        # only two spheres but error for all CV-folds
        self.failUnlessEqual(results.shape, (len(self.dataset.UC), 2))

        # test if we graciously puke if center_ids are out of bounds
        dataset0 = self.dataset[:, :50] # so we have no 50th feature
        self.failUnlessRaises(IndexError, sl, dataset0)

    def test_chi_square_searchlight(self):
        # only do partial to save time

        # Can't yet do this since test_searchlight isn't yet "under nose"
        #skip_if_no_external('scipy')
        if not externals.exists('scipy'):
            return

        from mvpa.misc.stats import chisquare

        transerror = TransferError(sample_clf_lin)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                enable_ca=['confusion'])


        def getconfusion(data):
            cv(data)
            return chisquare(cv.ca.confusion.matrix)[0]

        sl = sphere_searchlight(getconfusion, radius=0,
                         center_ids=[3,50])

        # run searchlight
        results = sl(self.dataset)
        self.failUnless(results.nfeatures == 2)



def suite():
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':
    import runner

