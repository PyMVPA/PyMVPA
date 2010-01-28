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
import numpy as N

from mvpa.base import cfg
# See other tests and test_procrust.py for some example on what to do ;)
from mvpa.algorithms.hyperalignment import Hyperalignment

# Somewhat slow but provides all needed ;)
from tests_warehouse import datasets, get_random_rotation, sweepargs

# if you need some classifiers
#from tests_warehouse_clfs import *

class HyperAlignmentTests(unittest.TestCase):


    @sweepargs(ref_ds=(None, 3))
    def test_basic_functioning(self, ref_ds):
        # get a dataset with some prominent trends in it
        ds4l = datasets['uni4large']
        # lets select for now only meaningful features
        ds_orig = ds4l[:, ds4l.a.nonbogus_features]
        n = 5 # # of datasets to generate
        Rs, dss_rotated, dss_rotated_clean, random_shifts, random_scales \
            = [], [], [], [], []
        # now lets compose derived datasets by using some random
        # rotation(s)
        for i in xrange(n):
            R = get_random_rotation(ds_orig.nfeatures)
            Rs.append(R)
            ds_ = ds_orig.copy()
            # reusing random data from dataset itself
            random_scales += [ds_orig.samples[i, 3] * 100]
            random_shifts += [ds_orig.samples[i+10] * 10]
            random_noise = ds4l.samples[:, ds4l.a.bogus_features[:4]]
            ds_.samples = N.dot(ds_orig.samples, R) * random_scales[-1] \
                          + random_shifts[-1]
            dss_rotated_clean.append(ds_)

            ds_ = ds_.copy()
            ds_.samples = ds_.samples + 0.1 * random_noise
            dss_rotated.append(ds_)

        ha = Hyperalignment(ref_ds=ref_ds)
        if ref_ds is None:
            ref_ds = 0                      # by default should be this one
        # Lets test two scenarios -- in one with no noise -- we should get
        # close to perfect reconstruction.  If noise was added -- not so good
        for noisy, dss in ((False, dss_rotated_clean),
                           (True, dss_rotated)):
            mappers = ha(dss)
            self.failUnlessEqual(ref_ds, ha.states.choosen_ref_ds)
            # Map data back

            dss_clean_back = [m.forward(ds_)
                              for m, ds_ in zip(mappers, dss_rotated_clean)]

            ds_norm = N.linalg.norm(dss[ref_ds].samples)
            nddss = []
            ds_orig_Rref = N.dot(ds_orig.samples, Rs[ref_ds]) \
                           * random_scales[ref_ds] \
                           + random_shifts[ref_ds]
            for ds_back in dss_clean_back:
                dds = ds_back.samples - ds_orig_Rref
                ndds = N.linalg.norm(dds) / ds_norm
                nddss += [ndds]
            if not noisy or cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnless(N.all(ndds <= (1e-10, 1e-2)[int(noisy)]),
                    msg="Should have reconstructed original dataset more or"
                        "less. Got normed differences %s in %s case."
                        % (nddss, ('clean', 'noisy')[int(noisy)]))

        # Lets see how well we do if asked to compute residuals
        ha = Hyperalignment(ref_ds=ref_ds, level2_niter=2,
                            enable_states=['residual_errors'])
        mappers = ha(dss_rotated_clean)
        self.failUnless(N.all(ha.states.residual_errors.sa.levels ==
                              ['1', '2:0', '2:1', '3']))
        rerrors = ha.states.residual_errors.samples
        # just basic tests:
        self.failUnlessEqual(rerrors[0, ref_ds], 0)
        self.failUnlessEqual(rerrors.shape, (4, n))
        pass


    def _testOnSwaroopData(self):
        #
        print "Running swaroops test on data we don't have"
        #from mvpa.datasets.miscfx import zscore
        #from mvpa.featsel.helpers import FixedNElementTailSelector
        #from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
        #   or just for lazy ones like yarik atm
        from mvpa.suite import *
        subj = ['cb', 'dm', 'hj', 'kd', 'kl', 'mh', 'ph', 'rb', 'se', 'sm']
        ds = []
        for sub in subj:
            ds.append(fmri_dataset(samples=sub+'_movie.nii.gz',
                                   mask=sub+'_mask_vt.nii.gz'))

        '''
        Compute feature ranks in each dataset
        based on correlation with other datasets
        '''
        feature_scores = [ N.zeros(ds[i].nfeatures) for i in range(len(subj)) ]
        '''
        for i in range(len(subj)):
            ds_temp = ds[i].samples - N.mean(ds[i].samples, axis=0)
            ds_temp = ds_temp / N.sqrt( N.sum( N.square(ds_temp), axis=0) )
            for j in range(i+1,len(subj)):
            ds_temp2 = ds[j].samples - N.mean(ds[j].samples, axis=0)
            ds_temp2 = ds_temp2 / N.sqrt( N.sum( N.square(ds_temp2), axis=0) )
            corr_temp= N.asarray(N.mat(N.transpose(ds_temp))*N.mat(ds_temp2))
            feature_scores[i] = feature_scores[i] + N.max(corr_temp, axis = 1)
            feature_scores[j] = feature_scores[j] + N.max(corr_temp, axis = 0)
        '''
        for i, sd in enumerate(ds):
            ds_temp = sd.copy()
            zscore(ds_temp, chunks=None)
            for j, sd2 in enumerate(ds[i+1:]):
                ds_temp2 = sd2.copy()
                zscore(ds_temp2, chunks=None)
                corr_temp = N.dot(ds_temp.samples.T, ds_temp2.samples)
                feature_scores[i] = feature_scores[i] + \
                                    N.max(corr_temp, axis = 1)
                feature_scores[j+i+1] = feature_scores[j+i+1] + \
                                        N.max(corr_temp, axis = 0)

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
                samples=sub+'_mkdg.nii.gz', labels=md_labels,
                chunks=N.repeat(range(8), 192), mask=sub+'_mask_vt.nii.gz'))

        m=mean_group_sample(['labels', 'chunks'])

        mkdg_ds = [ds_.get_mapped(m) for ds_ in mkdg_ds]
        mkdg_ds = [ds_[ds_.sa.labels != -1] for ds_ in mkdg_ds]
        [zscore(ds_, param_est=('labels', [0])) for ds_ in mkdg_ds]
        mkdg_ds = [ds_[ds_.sa.labels != 0] for ds_ in mkdg_ds]

        for i, sd in enumerate(mkdg_ds):
            sd.fa['bsc_scores'] = feature_scores[i]

        mkdg_ds_fs = [ sd[:, fselector(sd.fa.bsc_scores)] for sd in mkdg_ds]
        mkdg_ds_mapped = [ sd.get_mapped(mapper_results[i])
                           for i, sd in enumerate(mkdg_ds_fs)]

        # within-subject classification
        within_acc = []
        clf = clfswh['multiclass', 'linear', 'NU_SVC'][0]
        cvterr = CrossValidatedTransferError(
            TransferError(clf),
            splitter=NFoldSplitter(), enable_states=['confusion'])
        for sd in mkdg_ds_fs:
            wsc = cvterr(sd)
            within_acc.append(1-N.mean(wsc))

        within_acc_mapped = []
        for sd in mkdg_ds_mapped:
            wsc = cvterr(sd)
            within_acc_mapped.append(1-N.mean(wsc))

        print N.mean(within_acc)
        print N.mean(within_acc_mapped)

        mkdg_ds_all = vstack(mkdg_ds_mapped)
        mkdg_ds_all.sa['subject'] = N.repeat(range(10), 56)
        mkdg_ds_all.sa['chunks'] = mkdg_ds_all.sa['subject']

        bsc = cvterr(mkdg_ds_all)
        print 1-N.mean(bsc)
        mkdg_all = vstack(mkdg_ds_fs)
        mkdg_all.sa['chunks'] = N.repeat(range(10), 56)
        bsc_orig = cvterr(mkdg_all)
        print 1-N.mean(bsc_orig)
        pass



def suite():
    return unittest.makeSuite(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

