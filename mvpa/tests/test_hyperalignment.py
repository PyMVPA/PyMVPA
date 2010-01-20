# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ..."""

# See other tests and test_procrust.py for some example on what to do ;)
from mvpa.algorithms.hyperalignment import Hyperalignment

# Somewhat slow but provides all needed ;)
from tests_warehouse import *

# if you need some classifiers
#from tests_warehouse_clfs import *

class HyperAlignmentTests(unittest.TestCase):


    def testBasicFunctioning(self):
        # TODO
        pass

    def testPossibleInputs(self):
        # get a dataset with a very high SNR
        pass


    def _testOnSwaroopData(self):
        #
        print "Running swaroops test on data we don't have"
        subj = ['cb', 'dm', 'hj', 'kd', 'kl', 'mh', 'ph', 'rb', 'se', 'sm']
        ds = []
        for sub in subj: 
            ds.append(fmri_dataset(samples=sub+'_movie.nii.gz',mask=sub+'_mask_vt.nii.gz'))

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
            zscore(ds_temp, perchunk=False)
            for j, sd2 in enumerate(ds[i+1:]):
            ds_temp2 = sd2.copy()
            zscore(ds_temp2, perchunk=False)
            corr_temp= N.asarray(N.mat(N.transpose(ds_temp.samples))*N.mat(ds_temp2.samples))
            feature_scores[i] = feature_scores[i] + N.max(corr_temp, axis = 1)
            feature_scores[j+i+1] = feature_scores[j+i+1] + N.max(corr_temp, axis = 0)

        for i, sd in enumerate(ds):
            sd.fa['bsc_scores'] = feature_scores[i]

        fselector = FixedNElementTailSelector(2000, tail='upper', mode='select')

        ds_fs = [ sd[:, fselector(sd.fa.bsc_scores)] for sd in ds]

        hyper = Hyperalignment()
        mapper_results = hyper(datasets=ds_fs)

        md_cd = ColumnData('labels.txt',header=['label'])
        md_labels = [int(x) for x in md_cd['label']]
        for run in range(8):
            md_labels[192*run:192*run+3] = [-1]*3

        mkdg_ds = []
        for sub in subj:
            mkdg_ds.append(fmri_dataset(samples=sub+'_mkdg.nii.gz',labels=md_labels, chunks=N.repeat(range(8),192) , mask=sub+'_mask_vt.nii.gz'))

        m=mean_group_sample(['labels', 'chunks'])

        mkdg_ds = [mkdg_ds[i].get_mapped(m) for i in range(len(mkdg_ds))]
        mkdg_ds = [mkdg_ds[i][mkdg_ds[i].sa.labels != -1] for i in range(len(mkdg_ds))]
        [mkdg_ds[i].zscore(baselinelabels=0) for i in range(len(mkdg_ds))]
        mkdg_ds = [mkdg_ds[i][mkdg_ds[i].sa.labels != 0] for i in range(len(mkdg_ds))]

        for i, sd in enumerate(mkdg_ds):
            sd.fa['bsc_scores'] = feature_scores[i]

        mkdg_ds_fs = [ sd[:, fselector(sd.fa.bsc_scores)] for sd in mkdg_ds]
        mkdg_ds_mapped = [ sd.get_mapped(mapper_results[i]) for i, sd in enumerate(mkdg_ds_fs)]

        # within-subject classification
        within_acc = []
        clf = clfswh['multiclass','linear','NU_SVC'][0]
        cvterr = CrossValidatedTransferError(TransferError(clf), splitter=NFoldSplitter(), enable_states=['confusion'])
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
        mkdg_ds_all.sa['subject'] = N.repeat(range(10),56)
        mkdg_ds_all.sa['chunks'] = mkdg_ds_all.sa['subject']

        bsc = cvterr(mkdg_ds_all)
        print 1-N.mean(bsc)
        mkdg_all = vstack(mkdg_ds_fs)
        mkdg_all.sa['chunks'] = N.repeat(range(10),56)
        bsc_orig = cvterr(mkdg_all)
        print 1-N.mean(bsc_orig)
        pass



def suite():
    return unittest.makeSuite(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

