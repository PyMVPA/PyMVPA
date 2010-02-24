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
Classification of SVD-mapped Datasets
=====================================

.. index:: mapper, SVD, MappedClassifier

Demonstrate the usage of a dataset mapper performing data projection onto
singular value components within a cross-validation -- for *any* clasifier.
"""

from mvpa.suite import *

if __debug__:
    debug.active += ["CROSSC"]

#
# load PyMVPA example dataset
#
attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                        'attributes_literal.txt'))
dataset = fmri_dataset(os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       targets=attr.targets, chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

#
# preprocessing
#

# do chunkswise linear detrending on dataset
poly_detrend(dataset, polyord=1, chunks_attr='chunks')

# only use 'rest', 'cats' and 'scissors' samples from dataset
dataset = dataset[np.array([ l in ['rest', 'cat', 'scissors']
                    for l in dataset.targets], dtype='bool')]

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, chunks_attr='chunks', baselinetargets=['rest'], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset[dataset.sa.targets != 'rest']

# Specify the base classifier to be used
# To parametrize the classifier to be used
#   Clf = lambda *args:LinearCSVMC(C=-10, *args)
# Just to assign a particular classifier class
Clf = LinearCSVMC

# define some classifiers: a simple one and several classifiers with
# built-in SVDs
clfs = [('All orig.\nfeatures (%i)' % dataset.nfeatures, Clf()),
        ('All Comps\n(%i)' % (dataset.nsamples \
                 - (dataset.nsamples / len(dataset.UC)),),
                        MappedClassifier(Clf(), SVDMapper())),
        ('First 5\nComp.', MappedClassifier(Clf(),
                        SVDMapper(selector=range(5)))),
        ('First 30\nComp.', MappedClassifier(Clf(),
                        SVDMapper(selector=range(30)))),
        ('Comp.\n6-30', MappedClassifier(Clf(),
                        SVDMapper(selector=range(5,30))))]


# run and visualize in barplot
results = []
labels = []

for desc, clf in clfs:
    print desc
    cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_ca=['results'])
    cv(dataset)

    results.append(cv.ca.results)
    labels.append(desc)

plot_bars(results, labels=labels,
         title='Linear C-SVM classification (cats vs. scissors)',
         ylabel='Mean classification error (N-1 cross-validation, 12-fold)',
         distance=0.5)

if cfg.getboolean('examples', 'interactive', True):
    pl.show()

"""
Output of the example analysis:

.. image:: ../pics/ex_svdclf.*
   :align: center
   :alt: Generalization performance on the selected PCs.

"""
