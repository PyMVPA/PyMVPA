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
Sensitivity Measure
===================

.. index:: sensitivity

Run some basic and meta sensitivity measures on the example fMRI dataset that
comes with PyMVPA and plot the computed featurewise measures for each.  The
generated figure shows sensitivity maps computed by six sensitivity analyzers.

We start by loading PyMVPA and the example fMRI dataset.
"""

from mvpa.suite import *

# load PyMVPA example dataset
attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes.txt'))
dataset = NiftiDataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

"""As with classifiers it is easy to define a bunch of sensitivity
analyzers. It is usually possible to simply call `getSensitivityAnalyzer()`
on any classifier to get an instance of an appropriate sensitivity analyzer
that uses this particular classifier to compute and extract sensitivity scores.
"""

# define sensitivity analyzer
sensanas = {
    'a) ANOVA': OneWayAnova(transformer=N.abs),
    'b) Linear SVM weights': LinearNuSVMC().getSensitivityAnalyzer(
                                               transformer=N.abs),
    'c) I-RELIEF': IterativeRelief(transformer=N.abs),
    'd) Splitting ANOVA (odd-even)':
        SplitFeaturewiseMeasure(OneWayAnova(transformer=N.abs),
                                     OddEvenSplitter()),
    'e) Splitting SVM (odd-even)':
        SplitFeaturewiseMeasure(
            LinearNuSVMC().getSensitivityAnalyzer(transformer=N.abs),
                             OddEvenSplitter()),
    'f) I-RELIEF Online':
        IterativeReliefOnline(transformer=N.abs),
    'g) Splitting ANOVA (nfold)':
        SplitFeaturewiseMeasure(OneWayAnova(transformer=N.abs),
                                     NFoldSplitter()),
    'h) Splitting SVM (nfold)':
        SplitFeaturewiseMeasure(
            LinearNuSVMC().getSensitivityAnalyzer(transformer=N.abs),
                             NFoldSplitter()),
           }

"""Now, we are performing some a more or less standard preprocessing steps:
detrending, selecting a subset of the experimental conditions, normalization
of each feature to a standard mean and variance."""

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, model='linear')

# only use 'rest', 'shoe' and 'bottle' samples from dataset
dataset = dataset.selectSamples(
                N.array([ l in [0,3,7] for l in dataset.labels],
                dtype='bool'))

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=[0], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset.selectSamples(N.array([l != 0 for l in dataset.labels],
                                        dtype='bool'))

"""Finally, we will loop over all defined analyzers and let them compute
the sensitivity scores. The resulting vectors are then mapped back into the
dataspace of the original fMRI samples, which are then plotted."""

fig = 0
P.figure(figsize=(14, 8))

keys = sensanas.keys()
keys.sort()

for s in keys:
    # tell which one we are doing
    print "Running %s ..." % (s)

    # compute sensitivies
    # I-RELIEF assigns zeros, which corrupts voxel masking for pylab's
    # imshow, so adding some epsilon :)
    smap = sensanas[s](dataset)+0.00001

    # map sensitivity map into original dataspace
    orig_smap = dataset.mapReverse(smap)
    masked_orig_smap = N.ma.masked_array(orig_smap, mask=orig_smap == 0)

    # make a new subplot for each classifier
    fig += 1
    P.subplot(3, 3, fig)

    P.title(s)

    P.imshow(masked_orig_smap[0],
             interpolation='nearest',
             aspect=1.25,
             cmap=P.cm.autumn)

    # uniform scaling per base sensitivity analyzer
    if s.count('ANOVA'):
        P.clim(0, 30)
    elif s.count('SVM'):
        P.clim(0, 0.055)
    else:
        pass

    P.colorbar(shrink=0.6)

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()

"""
Output of the example analysis:

.. image:: ../pics/ex_sensanas.*
   :align: center
   :alt: Various sensitivity analysis results

"""
