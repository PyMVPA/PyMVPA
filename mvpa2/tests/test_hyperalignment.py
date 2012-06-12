# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ..."""

import unittest
import numpy as np

from mvpa2.base import cfg
from mvpa2.datasets.base import Dataset

# See other tests and test_procrust.py for some example on what to do ;)
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.support import idhash
from mvpa2.misc.data_generators import random_affine_transformation
from mvpa2.misc.fx import get_random_rotation

# Somewhat slow but provides all needed ;)
from mvpa2.testing import sweepargs, reseed_rng
from mvpa2.testing.datasets import datasets

from mvpa2.generators.partition import NFoldPartitioner

# if you need some classifiers
#from mvpa2.testing.clfs import *

class HyperAlignmentTests(unittest.TestCase):


    @sweepargs(zscore_all=(False, True))
    @sweepargs(zscore_common=(False, True))
    @sweepargs(ref_ds=(None, 2))
    @reseed_rng()
    def test_basic_functioning(self, ref_ds, zscore_common, zscore_all):
        ha = Hyperalignment(ref_ds=ref_ds,
                            zscore_all=zscore_all,
                            zscore_common=zscore_common)
        if ref_ds is None:
            ref_ds = 0                      # by default should be this one

        # get a dataset with some prominent trends in it
        ds4l = datasets['uni4large']
        # lets select for now only meaningful features
        ds_orig = ds4l[:, ds4l.a.nonbogus_features]
        nf = ds_orig.nfeatures
        n = 4 # # of datasets to generate
        Rs, dss_rotated, dss_rotated_clean, random_shifts, random_scales \
            = [], [], [], [], []

        # now lets compose derived datasets by using some random
        # rotation(s)
        for i in xrange(n):
            ## if False: # i == ref_ds:
            #     # Do not rotate the target space so we could check later on
            #     # if we transform back nicely
            #     R = np.eye(ds_orig.nfeatures)
            ## else:
            ds_ = random_affine_transformation(ds_orig, scale_fac=100, shift_fac=10)
            Rs.append(ds_.a.random_rotation)
            # reusing random data from dataset itself
            random_scales += [ds_.a.random_scale]
            random_shifts += [ds_.a.random_shift]
            random_noise = ds4l.samples[:, ds4l.a.bogus_features[:4]]

            ## if (zscore_common or zscore_all):
            ##     # for later on testing of "precise" reconstruction
            ##     zscore(ds_, chunks_attr=None)

            dss_rotated_clean.append(ds_)

            ds_ = ds_.copy()
            ds_.samples = ds_.samples + 0.1 * random_noise
            dss_rotated.append(ds_)

        # Lets test two scenarios -- in one with no noise -- we should get
        # close to perfect reconstruction.  If noise was added -- not so good
        for noisy, dss in ((False, dss_rotated_clean),
                           (True, dss_rotated)):
            # to verify that original datasets didn't get changed by
            # Hyperalignment store their idhashes of samples
            idhashes = [idhash(ds.samples) for ds in dss]
            idhashes_targets = [idhash(ds.targets) for ds in dss]

            mappers = ha(dss)

            idhashes_ = [idhash(ds.samples) for ds in dss]
            idhashes_targets_ = [idhash(ds.targets) for ds in dss]
            self.assertEqual(idhashes, idhashes_,
                msg="Hyperalignment must not change original data.")
            self.assertEqual(idhashes_targets, idhashes_targets_,
                msg="Hyperalignment must not change original data targets.")

            self.assertEqual(ref_ds, ha.ca.choosen_ref_ds)

            # Map data back

            dss_clean_back = [m.forward(ds_)
                              for m, ds_ in zip(mappers, dss_rotated_clean)]

            ds_norm = np.linalg.norm(dss[ref_ds].samples)
            nddss = []
            ndcss = []
            ds_orig_Rref = np.dot(ds_orig.samples, Rs[ref_ds]) \
                           * random_scales[ref_ds] \
                           + random_shifts[ref_ds]
            if zscore_common or zscore_all:
                zscore(Dataset(ds_orig_Rref), chunks_attr=None)
            for ds_back in dss_clean_back:
                # if we used zscoring of common, we cannot rely
                # that range/offset could be matched, so lets use
                # corrcoef
                ndcs = np.diag(np.corrcoef(ds_back.samples.T,
                                           ds_orig_Rref.T)[nf:, :nf], k=0)
                ndcss += [ndcs]
                dds = ds_back.samples - ds_orig_Rref
                ndds = np.linalg.norm(dds) / ds_norm
                nddss += [ndds]
            snoisy = ('clean', 'noisy')[int(noisy)]
            do_labile = cfg.getboolean('tests', 'labile', default='yes')
            if not noisy or do_labile:
                # First compare correlations
                self.assertTrue(np.all(np.array(ndcss)
                                       >= (0.9, 0.85)[int(noisy)]),
                        msg="Should have reconstructed original dataset more or"
                        " less. Got correlations %s in %s case."
                        % (ndcss, snoisy))
                if not (zscore_all or zscore_common):
                    # if we didn't zscore -- all of them should be really close
                    self.assertTrue(np.all(np.array(nddss)
                                       <= (1e-10, 1e-1)[int(noisy)]),
                        msg="Should have reconstructed original dataset well "
                        "without zscoring. Got normed differences %s in %s case."
                        % (nddss, snoisy))
                elif do_labile:
                    # otherwise they all should be somewhat close
                    #print snoisy, ref_ds,  nddss
                    self.assertTrue(np.all(np.array(nddss) >= nddss[ref_ds]),
                        msg="Should have reconstructed orig_ds best of all. "
                        "Got normed differences %s in %s case with ref_ds=%d."
                        % (nddss, snoisy, ref_ds))
                    self.assertTrue(np.all(np.array(nddss)
                                           <= (.2, 3)[int(noisy)]),
                        msg="Should have reconstructed original dataset more or"
                        " less for all. Got normed differences %s in %s case."
                        % (nddss, snoisy))
                    self.assertTrue(np.all(nddss[ref_ds] <= .05),
                        msg="Should have reconstructed original dataset quite "
                        "well even with zscoring. Got normed differences %s "
                        "in %s case." % (nddss, snoisy))

        # Lets see how well we do if asked to compute residuals
        ha = Hyperalignment(ref_ds=ref_ds, level2_niter=2,
                            enable_ca=['residual_errors'])
        mappers = ha(dss_rotated_clean)
        self.assertTrue(np.all(ha.ca.residual_errors.sa.levels ==
                              ['1', '2:0', '2:1', '3']))
        rerrors = ha.ca.residual_errors.samples
        # just basic tests:
        self.assertEqual(rerrors[0, ref_ds], 0)
        self.assertEqual(rerrors.shape, (4, n))


    def _test_on_swaroop_data(self):
        #
        print "Running swaroops test on data we don't have"
        #from mvpa2.datasets.miscfx import zscore
        #from mvpa2.featsel.helpers import FixedNElementTailSelector
        #   or just for lazy ones like yarik atm
        #enable to test from mvpa2.suite import *
        subj = ['cb', 'dm', 'hj', 'kd', 'kl', 'mh', 'ph', 'rb', 'se', 'sm']
        ds = []
        for sub in subj:
            ds.append(fmri_dataset(samples=sub+'_movie.nii.gz',
                                   mask=sub+'_mask_vt.nii.gz'))

        '''
        Compute feature ranks in each dataset
        based on correlation with other datasets
        '''
        feature_scores = [ np.zeros(ds[i].nfeatures) for i in range(len(subj)) ]
        '''
        for i in range(len(subj)):
            ds_temp = ds[i].samples - np.mean(ds[i].samples, axis=0)
            ds_temp = ds_temp / np.sqrt( np.sum( np.square(ds_temp), axis=0) )
            for j in range(i+1,len(subj)):
            ds_temp2 = ds[j].samples - np.mean(ds[j].samples, axis=0)
            ds_temp2 = ds_temp2 / np.sqrt( np.sum( np.square(ds_temp2), axis=0) )
            corr_temp= np.asarray(np.mat(np.transpose(ds_temp))*np.mat(ds_temp2))
            feature_scores[i] = feature_scores[i] + np.max(corr_temp, axis = 1)
            feature_scores[j] = feature_scores[j] + np.max(corr_temp, axis = 0)
        '''
        for i, sd in enumerate(ds):
            ds_temp = sd.copy()
            zscore(ds_temp, chunks_attr=None)
            for j, sd2 in enumerate(ds[i+1:]):
                ds_temp2 = sd2.copy()
                zscore(ds_temp2, chunks_attr=None)
                corr_temp = np.dot(ds_temp.samples.T, ds_temp2.samples)
                feature_scores[i] = feature_scores[i] + \
                                    np.max(corr_temp, axis = 1)
                feature_scores[j+i+1] = feature_scores[j+i+1] + \
                                        np.max(corr_temp, axis = 0)

        for i, sd in enumerate(ds):
            sd.fa['bsc_scores'] = feature_scores[i]

        fselector = FixedNElementTailSelector(2000,
                                              tail='upper', mode='select')

        ds_fs = [ sd[:, fselector(sd.fa.bsc_scores)] for sd in ds]

        hyper = Hyperalignment()
        mapper_results = hyper(datasets=ds_fs)

        md_cd = ColumnData('labels.txt', header=['label'])
        md_labels = [int(x) for x in md_cd['label']]
        for run in range(8):
            md_labels[192*run:192*run+3] = [-1]*3

        mkdg_ds = []
        for sub in subj:
            mkdg_ds.append(fmri_dataset(
                samples=sub+'_mkdg.nii.gz', targets=md_labels,
                chunks=np.repeat(range(8), 192), mask=sub+'_mask_vt.nii.gz'))

        m = mean_group_sample(['targets', 'chunks'])

        mkdg_ds = [ds_.get_mapped(m) for ds_ in mkdg_ds]
        mkdg_ds = [ds_[ds_.sa.targets != -1] for ds_ in mkdg_ds]
        [zscore(ds_, param_est=('targets', [0])) for ds_ in mkdg_ds]
        mkdg_ds = [ds_[ds_.sa.targets != 0] for ds_ in mkdg_ds]

        for i, sd in enumerate(mkdg_ds):
            sd.fa['bsc_scores'] = feature_scores[i]

        mkdg_ds_fs = [ sd[:, fselector(sd.fa.bsc_scores)] for sd in mkdg_ds]
        mkdg_ds_mapped = [ sd.get_mapped(mapper_results[i])
                           for i, sd in enumerate(mkdg_ds_fs)]

        # within-subject classification
        within_acc = []
        clf = clfswh['multiclass', 'linear', 'NU_SVC'][0]
        cvterr = CrossValidation(clf, NFoldPartitioner(),
                                 enable_ca=['confusion'])
        for sd in mkdg_ds_fs:
            wsc = cvterr(sd)
            within_acc.append(1-np.mean(wsc))

        within_acc_mapped = []
        for sd in mkdg_ds_mapped:
            wsc = cvterr(sd)
            within_acc_mapped.append(1-np.mean(wsc))

        print np.mean(within_acc)
        print np.mean(within_acc_mapped)

        mkdg_ds_all = vstack(mkdg_ds_mapped)
        mkdg_ds_all.sa['subject'] = np.repeat(range(10), 56)
        mkdg_ds_all.sa['chunks'] = mkdg_ds_all.sa['subject']

        bsc = cvterr(mkdg_ds_all)
        print 1-np.mean(bsc)
        mkdg_all = vstack(mkdg_ds_fs)
        mkdg_all.sa['chunks'] = np.repeat(range(10), 56)
        bsc_orig = cvterr(mkdg_all)
        print 1-np.mean(bsc_orig)
        pass



def suite():
    return unittest.makeSuite(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

