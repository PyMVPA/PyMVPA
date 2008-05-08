#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating a how to use data projection onto PCA components
for *any* clasifier"""

import numpy as N
import pylab as P

# local imports
from mvpa.misc.iohelpers import SampleAttributes
from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.datasets.misc import zscore
from mvpa.misc.signal import detrend
from mvpa.clfs.transerror import TransferError
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.svm import LinearCSVMC
from mvpa.clfs.classifier import MappedClassifier
from mvpa.mappers import PCAMapper

from mvpa.misc import debug
debug.active += ["CROSSC"]


# plotting helper function
def makeBarPlot(data, labels=None, title=None, ylim=None, ylabel=None):
    xlocations = N.array(range(len(data))) + 0.5
    width = 0.5

    # work with arrays
    data = N.array(data)

    # plot bars
    plot = P.bar(xlocations,
                 data.mean(axis=1),
                 yerr=data.std(axis=1) / N.sqrt(data.shape[1]),
                 width=width,
                 color='0.6',
                 ecolor='black')
    P.axhline(0.5, ls='--', color='0.4')

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xlocations+ width/2, labels)

    if ylabel:
        P.ylabel(ylabel)

    P.xlim(0, xlocations[-1]+width*2)


#
# load PyMVPA example dataset
#
attr = SampleAttributes('data/attributes.txt')
dataset = NiftiDataset(samples='data/bold.nii.gz',
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask='data/mask.nii.gz')

#
# preprocessing
#

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, model='linear')

# only use 'rest', 'face' and 'house' samples from dataset
dataset = dataset.selectSamples(
                N.array([ l in [0,4,5] for l in dataset.labels], dtype='bool'))

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=[0], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset.selectSamples(N.array([l != 0 for l in dataset.labels],
                                        dtype='bool'))

# define some classifiers: a simple one and several classifiers with built-in
# PCAs
clfs = [('All orig. features', LinearCSVMC()),
        ('All PCs', MappedClassifier(LinearCSVMC(), PCAMapper())),
        ('First 3 PCs', MappedClassifier(LinearCSVMC(),
                        PCAMapper(selector=range(5)))),
        ('First 50 PCs', MappedClassifier(LinearCSVMC(),
                        PCAMapper(selector=range(50)))),
        ('PCs 3-50', MappedClassifier(LinearCSVMC(),
                        PCAMapper(selector=range(3,50))))]


# run and visualize in barplot
results = []
labels = []

for desc, clf in clfs:
    print desc
    cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_states=['results'])
    cv(dataset)

    results.append(cv.results)
    labels.append(desc)

makeBarPlot(results,labels=labels, title='Linear C-SVM classification')
P.show()
