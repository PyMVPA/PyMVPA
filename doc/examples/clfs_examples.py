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

from mvpa2.suite import *

# no MVPA warnings during this example
warning.handlers = []

def main():

    # fix seed or set to None for new each time
    np.random.seed(44)


    # Load Haxby dataset example
    haxby8 = load_example_fmri_dataset(literal=True)
    haxby8.samples = haxby8.samples.astype(np.float32)

    # preprocess slightly
    poly_detrend(haxby8, chunks_attr='chunks', polyord=1)
    zscore(haxby8, chunks_attr='chunks', param_est=('targets', 'rest'))

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
            # Let's prevent failures of the entire script if some particular
            # classifier is not appropriate for the data
            try:
                # Change to False if you want to use CrossValidation
                # helper, instead of going through splits manually to
                # track training/prediction time of the classifiers
                do_explicit_splitting = True
                if not do_explicit_splitting:
                    cv = CrossValidation(
                        clf, NFoldPartitioner(), enable_ca=['stats', 'calling_time'])
                    error = cv(dataset)
                    # print cv.ca.stats
                    print "%5.1f%%      -    \t   -       -    %.2fs" \
                          % (cv.ca.stats.percent_correct, cv.ca.calling_time)
                    continue

                # To report transfer error (and possibly some other metrics)
                confusion = ConfusionMatrix()
                times = []
                nf = []
                t0 = time.time()
                #TODO clf.ca.enable('nfeatures')
                partitioner = NFoldPartitioner()
                for nfold, ds in enumerate(partitioner.generate(dataset)):
                    (training_ds, validation_ds) = tuple(
                        Splitter(attr=partitioner.space).generate(ds))
                    clf.train(training_ds)
                    #TODO nf.append(clf.ca.nfeatures)
                    predictions = clf.predict(validation_ds.samples)
                    confusion.add(validation_ds.targets, predictions)
                    times.append([clf.ca.training_time, clf.ca.predicting_time])

                tfull = time.time() - t0
                times = np.mean(times, axis=0)
                #TODO nf = np.mean(nf)
                # print confusion
                #TODO print "%5.1f%%   %-4d\t %.2fs  %.2fs   %.2fs" % \
                print "%5.1f%%       -   \t %.2fs  %.2fs   %.2fs" % \
                      (confusion.percent_correct, times[0], times[1], tfull)
                #TODO      (confusion.percent_correct, nf, times[0], times[1], tfull)
            except LearnerError, e:
                print " skipped due to '%s'" % e

if __name__ == "__main__":
    main()
