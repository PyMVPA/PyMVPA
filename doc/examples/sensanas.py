#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating some SensitivityAnalyzers performing on a fMRI
dataset with brain activity recorded while perceiving images of objects
(shoes vs. chairs).

Generated images show sensitivity maps computed by six sensitivity analyzers.

This example assumes that the PyMVPA example dataset is located in data/.
"""

import numpy as N
import pylab as P

# local imports
from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.misc.iohelpers import SampleAttributes
from mvpa.algorithms.anova import OneWayAnova
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.algorithms.linsvmweights import LinearSVMWeights
from mvpa.datasets.misc import zscore
from mvpa.misc.signal import detrend
from mvpa.algorithms.splitsensana import SplittingSensitivityAnalyzer
from mvpa.datasets.splitter import OddEvenSplitter, NFoldSplitter

# load PyMVPA example dataset
attr = SampleAttributes('data/attributes.txt')
dataset = NiftiDataset(samples='data/bold.nii.gz',
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask='data/mask.nii.gz')

# define sensitivity analyzer
sensanas = {'a) ANOVA': OneWayAnova(transformer=N.abs),
            'b) Linear SVM weights': LinearSVMWeights(LinearNuSVMC(),
                                                   transformer=N.abs),
            'c) Splitting ANOVA (odd-even)':
                SplittingSensitivityAnalyzer(OneWayAnova(transformer=N.abs),
                                             OddEvenSplitter()),
            'd) Splitting SVM (odd-even)':
                SplittingSensitivityAnalyzer(
                    LinearSVMWeights(LinearNuSVMC(), transformer=N.abs),
                                     OddEvenSplitter()),
            'e) Splitting ANOVA (nfold)':
                SplittingSensitivityAnalyzer(OneWayAnova(transformer=N.abs),
                                             NFoldSplitter()),
            'f) Splitting SVM (nfold)':
                SplittingSensitivityAnalyzer(
                    LinearSVMWeights(LinearNuSVMC(), transformer=N.abs),
                                     NFoldSplitter())
           }

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, type='linear')

# only use 'rest', 'shoe' and 'bottle' samples from dataset
dataset = dataset.selectSamples(
                N.array([ l in [0,3,7] for l in dataset.labels], dtype='bool'))

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=[0], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset.selectSamples(N.array([l != 0 for l in dataset.labels],
                                        dtype='bool'))

fig = 0
P.figure(figsize=(8,8))

keys = sensanas.keys()
keys.sort()

for s in keys:
    # tell which one we are doing
    print "Running %s ..." % (s)

    # compute sensitivies
    smap = sensanas[s](dataset)

    # map sensitivity map into original dataspace
    orig_smap = dataset.mapReverse(smap)
    masked_orig_smap = N.ma.masked_array(orig_smap, mask=orig_smap == 0)

    # make a new subplot for each classifier
    fig += 1
    P.subplot(3,2,fig)

    P.title(s)

    P.imshow(masked_orig_smap[0],
             interpolation='nearest',
             aspect=1.25,
             cmap=P.cm.autumn)

    # uniform scaling per base sensitivity analyzer
    if s.count('ANOVA'):
        P.clim(0, 0.4)
    else:
        P.clim(0, 0.055)

    P.colorbar(shrink=0.6)


# show all the cool figures
P.show()
