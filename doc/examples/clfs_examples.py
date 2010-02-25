#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Classifier Sweep
================

This examples shows a test of various classifiers on different datasets.
"""

from mvpa.suite import *

# no MVPA warnings during whole testsuite
warning.handlers = []

def main():

    # fix seed or set to None for new each time
    np.random.seed(44)


    # Load Haxby dataset example
    attrs = SampleAttributes(os.path.join(pymvpa_dataroot,
                                          'attributes_literal.txt'))
    haxby8 = fmri_dataset(samples=os.path.join(pymvpa_dataroot,
                                               'bold.nii.gz'),
                          targets=attrs.targets,
                          chunks=attrs.chunks,
                          mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))
    haxby8.samples = haxby8.samples.astype(np.float32)

    # preprocess slightly
    detrend(haxby8, chunks_attr='chunks', model='linear')
    zscore(haxby8, chunks_attr='chunks', baselinetargets=['rest'],
           targetdtype='float32')
    haxby8_no0 = haxby8[haxby8.targets != 'rest']

    dummy2 = normal_feature_dataset(perlabel=30, nlabels=2,
                                  nfeatures=100,
                                  nchunks=6, nonbogus_features=[11, 10],
                                  snr=3.0)

    for (dataset, datasetdescr), clfs_ in \
        [
        ((dummy2,
          "Dummy 2-class univariate with 2 useful features out of 100"),
          clfswh[:]),
        ((pure_multivariate_signal(8, 3),
          "Dummy XOR-pattern"),
          clfswh['non-linear']),
        ((haxby8_no0,
          "Haxby 8-cat subject 1"),
          clfswh['multiclass']),
        ]:
        # XXX put back whenever there is summary() again
        #print "%s\n %s" % (datasetdescr, dataset.summary(idhash=False))
        print " Classifier on %s\n" \
                "                                          :   %%corr   " \
                "#features\t train  predict full" % datasetdescr
        for clf in clfs_:
            print "  %-40s: "  % clf.descr,
            # Lets do splits/train/predict explicitely so we could track
            # timing otherwise could be just
            #cv = CrossValidatedTransferError(
            #         TransferError(clf),
            #         NFoldSplitter(),
            #         enable_ca=['confusion'])
            #error = cv(dataset)
            #print cv.confusion

            # to report transfer error
            confusion = ConfusionMatrix()
            times = []
            nf = []
            t0 = time.time()
            clf.ca.enable('feature_ids')
            for nfold, (training_ds, validation_ds) in \
                    enumerate(NFoldSplitter()(dataset)):
                clf.train(training_ds)
                nf.append(len(clf.ca.feature_ids))
                if nf[-1] == 0:
                    break
                predictions = clf.predict(validation_ds.samples)
                confusion.add(validation_ds.targets, predictions)
                times.append([clf.ca.training_time, clf.ca.predicting_time])
            if nf[-1] == 0:
                print "no features were selected. skipped"
                continue
            tfull = time.time() - t0
            times = np.mean(times, axis=0)
            nf = np.mean(nf)
            # print "\n", confusion
            print "%5.1f%%   %-4d\t %.2fs  %.2fs   %.2fs" % \
                  (confusion.percent_correct, nf, times[0], times[1], tfull)


if __name__ == "__main__":
    main()
