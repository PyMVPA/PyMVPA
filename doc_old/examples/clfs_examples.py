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
    N.random.seed(44)


    # Load Haxby dataset example
    attrs = SampleAttributes(os.path.join(pymvpa_dataroot,
                                          'attributes_literal.txt'))
    haxby8 = fmri_dataset(samples=os.path.join(pymvpa_dataroot,
                                               'bold.nii.gz'),
                          labels=attrs.labels,
                          chunks=attrs.chunks,
                          mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))
    haxby8.samples = haxby8.samples.astype(N.float32)

    # preprocess slightly
    detrend(haxby8, perchunk=True, model='linear')
    zscore(haxby8, perchunk=True, baselinelabels=['rest'],
           targetdtype='float32')
    haxby8_no0 = haxby8[haxby8.labels != 'rest']

    dummy2 = normalFeatureDataset(perlabel=30, nlabels=2,
                                  nfeatures=100,
                                  nchunks=6, nonbogus_features=[11, 10],
                                  snr=3.0)

    for (dataset, datasetdescr), clfs_ in \
        [
        ((dummy2,
          "Dummy 2-class univariate with 2 useful features out of 100"),
          clfswh[:]),
        ((pureMultivariateSignal(8, 3),
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
            #         enable_states=['confusion'])
            #error = cv(dataset)
            #print cv.confusion

            # to report transfer error
            confusion = ConfusionMatrix()#labels_map=dataset.labels_map)
            times = []
            nf = []
            t0 = time.time()
            clf.states.enable('feature_ids')
            for nfold, (training_ds, validation_ds) in \
                    enumerate(NFoldSplitter()(dataset)):
                clf.train(training_ds)
                nf.append(len(clf.states.feature_ids))
                if nf[-1] == 0:
                    break
                predictions = clf.predict(validation_ds.samples)
                confusion.add(validation_ds.labels, predictions)
                times.append([clf.states.training_time, clf.states.predicting_time])
            if nf[-1] == 0:
                print "no features were selected. skipped"
                continue
            tfull = time.time() - t0
            times = N.mean(times, axis=0)
            nf = N.mean(nf)
            # print "\n", confusion
            print "%5.1f%%   %-4d\t %.2fs  %.2fs   %.2fs" % \
                  (confusion.percentCorrect, nf, times[0], times[1], tfull)


if __name__ == "__main__":
    main()
