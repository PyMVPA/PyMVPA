#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating some FeaturewiseDatasetMeasures performing on a fMRI
dataset with brain activity recorded while perceiving images of objects
(shoes vs. chairs).

Generated images show sensitivity maps computed by six sensitivity analyzers.

This example assumes that the PyMVPA example dataset is located in data/.
"""

from mvpa.suite import *

# load PyMVPA example dataset
attr = SampleAttributes('data/attributes.txt')
dataset = NiftiDataset(samples='data/bold.nii.gz',
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask='data/mask.nii.gz')

# define sensitivity analyzer
sensanas = {'a) ANOVA': OneWayAnova(transformer=N.abs),
            'b) Linear SVM weights': LinearNuSVMC().getSensitivityAnalyzer(
                                                       transformer=N.abs),
            'c) Splitting ANOVA (odd-even)':
                SplitFeaturewiseMeasure(OneWayAnova(transformer=N.abs),
                                             OddEvenSplitter()),
            'd) Splitting SVM (odd-even)':
                SplitFeaturewiseMeasure(
                    LinearNuSVMC().getSensitivityAnalyzer(transformer=N.abs),
                                     OddEvenSplitter()),
            'e) Splitting ANOVA (nfold)':
                SplitFeaturewiseMeasure(OneWayAnova(transformer=N.abs),
                                             NFoldSplitter()),
            'f) Splitting SVM (nfold)':
                SplitFeaturewiseMeasure(
                    LinearNuSVMC().getSensitivityAnalyzer(transformer=N.abs),
                                     NFoldSplitter())
           }

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, model='linear')

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


if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()
