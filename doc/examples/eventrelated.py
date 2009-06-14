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
Spatio-temporal Analysis of event-related fMRI data
===================================================

.. index:: event-related fMRI

The purpose of this example is to show how to use spatio-temporal
samples in an event-related fMRI data analysis. We start as usual by
loading the PyMVPA suite. The tiny fMRI dataset, included in the sources
will server as an example dataset.  Although the original paradigm of
this dataset is a block-design experiment, we'll analyze it in an
event-related fashion, where each block will be considered as an
individual event.

"""

from mvpa.suite import *

# filename of the source fMRI timeseries image
fmri_src = os.path.join(pymvpa_dataroot, 'bold.nii.gz')

mask = NiftiImage(os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

# load the samples attributes as usual and preserve the
# literal labels
attr = SampleAttributes(
                os.path.join(pymvpa_dataroot,
                             'attributes_literal.txt'),
                        literallabels=True)

"""

For an event-related analysis most of the processing is done on data
samples that are somehow derived from a set of events. The rest of the
data could be considered irrelevant. However, some e.g. preprocessing is
only meaningful when performed on the full timeseries and not the
segmented event samples. An example is the detrending that
typically needs to be done on the original, continuous timeseries.
Therefore we are going to load the data twice: first as a simple
volume-based dataset for an initial preprocessing, and second to extract
the events of interest.

"""

verbose(1, "Load data for preprocessing")
pre_ds = NiftiImage(fmri_src)

# actual labels are not important here, could be 'labels=1'
pre_ds = NiftiDataset(samples=fmri_src, labels=attr.labels,
                      chunks=attr.chunks, mask=mask)

# convert to floats
pre_ds.setSamplesDType('float')

# detrend on full timeseries
detrend(pre_ds, perchunk=True, model='linear')

"""

After the detrending, we can now segment the timeseries into a set of
events. To achieve this we have to compile a list of event definitions
first. In this example we will simply convert the block-design setup
defined by the samples attributes into events, so that each block become
an event with an associated onset and duration. The necessary chunk
settings are taken from these attributes as well. Finally, we are only
interested in `face` or `house` blocks.

"""

evs = [ev for ev in attr.toEvents()
            if ev['label'] in ['face', 'house']]

"""

Since we might want to take a look at the sensitivity profile ranging
from just before until a little after each block, we are slightly moving
the event onsets (one volume prior the actual event) and request to
extract a set of twelve consecutive volume a as sample for each event.

"""

for ev in evs:
    ev['onset'] -= 1
    ev['duration'] = 12

"""

A :class:`~mvpa.datasets.nifti.ERNiftiDataset` can now be used to
segment the timeseries and automatically extract boxcar-shaped multi-volume
samples. It is also capable of applying a volume mask.

"""

# could use evconv...
verbose(1, "Segmenting timeseries into events")
ds = ERNiftiDataset(samples=pre_ds.map2Nifti(),
                    events=evs,
                    mask=mask,
                    labels_map={'face': 1,
                                'house': 2})

"""

For demonstration purposes we copy the pristine dataset before any
further processing is done.

"""

# preserve
orig_ds = deepcopy(ds)

"""

The rest is pretty much standard. A dataset with spatio-temporal fMRI
samples behaves just as any other dataset type. We perform normalization
by Z-scoring the data and settle on a linear SVM classifier to perform a
cross-validated sensitivity analysis.

"""

# using rest as baseline
zscore(ds, perchunk=True)

clf = LinearCSVMC()
sclf = SplitClassifier(clf, NFoldSplitter(),
       enable_states=['confusion', 'training_confusion'])

# Compute sensitivity, which in turn trains the sclf
sensitivities = \
    sclf.getSensitivityAnalyzer(combiner=None,
                                slave_combiner=None)(ds)

"""

Before looking at the sensitivity profile we first have to inspect the
classifier performance in the cross-validation, since only for a model
with reasonable generalization performance it would make sense to
interpret the model parameters, i.e. classifier weights. If this is done
we could dump the spatio-temporal sensitivity profile, which covers all
voxels in the dataset for the full duration of the events, into a NIfTI
file.

"""

print sclf.confusion

#ds.map2Nifti(N.mean(sensitivities, axis=0)).save('fs_sens.nii.gz')

"""

However, we are going to plot it for some target voxel right away, and
compare it to the actual signal timecourse prior and after
normalization. We can use the dataset's mapper to convert the
sensitivity vector for each CV-fold back into a 4D snippet.

"""

# reverse map sensitivities -> fold x volumes x Z x Y x X
smaps = N.array([ds.mapReverse(s) for s in sensitivities])

# extract sensitivity profile for target voxel ijk(33,10,0)
v = (0, 3, 15)
smap = smaps[:,:,v[0],v[1],v[2]]

"""

Now, we plot the orginal signal after initial detrending,

"""

P.subplot(311)
P.title('Voxel zyx%s\nblock-onset@1, block-offset@8' % `v`)
for l in ds.uniquelabels:
    P.plot(
        ds.mapReverse(
            orig_ds.samples[ds.labels==l].mean(axis=0)
                )[:,v[0],v[1],v[2]])
P.ylabel('Signal after detrending')
P.axhline(linestyle='--', color='0.6')

"""

the peristimulus timecourse after Z-scoring,

"""

P.subplot(312)
for l in ds.uniquelabels:
    P.plot(
        ds.mapReverse(
            ds.samples[ds.labels==l].mean(axis=0)
                )[:,v[0],v[1],v[2]])
P.ylabel('Signal after normalization')
P.axhline(linestyle='--', color='0.6')

"""

and finally the associated SVM weight profile for each peristimulus
timepoint of the voxel.

"""

P.subplot(313)
plotErrLine(smap)
P.ylabel('Sensitivity')
P.xlabel('Peristimulus volumes')
P.axhline(linestyle='--', color='0.6')


if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()

