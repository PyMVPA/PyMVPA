.. _datadb_tutorial_data:

********************************
Tutorial Data: Block-design fMRI
********************************

This dataset is a compilation of data and results for :ref:`PyMVPA
Tutorial <chap_tutorial>`.

At the moment dataset is based on data for a single subject from a study published by :ref:`Haxby
et al. (2001) <HGF+01>`. The full (raw) dataset of this study is also
:ref:`available <datadb_haxby2001>`. However, in constrast to the full data
this single subject datasets has been preprocessed to a degree that should
allow people without prior fMRI experience to perform meaningful analyses.
Moreover, it should not require further preprocessing with external tools.

All preprocessing has been performed using tools from FSL_. Specifically, the
4D fMRI timeseries has been motion-corrected by applying MCFLIRT to a
skull-stripped and thresholded timeseries (to zero-out non-brain voxels,
using a brain outline estimate significantly larger than the brain, to
prevent removal of edge voxels actually covering brain tissue). The
estimated motion parameters have been subsequently applied to the original
(unthresholded, unstripped) timeseries. For simplicity the T1-weighed
anatomical image has also been projected and resampled into the subjects
functional space.


Terms Of Use
============

The orginal authors of :ref:`Haxby et al. (2001) <HGF+01>` hold the copyright
of this dataset and made it available under the terms of the `Creative Commons
Attribution-Share Alike 3.0`_ license. The PyMVPA authors have preprocessed the
data and released this derivative work under the same licensing terms.

.. _Creative Commons Attribution-Share Alike 3.0: http://creativecommons.org/licenses/by-sa/3.0/


Download
========

A tarball is available at:

  http://data.pymvpa.org/datasets/tutorial_data


Tarball Content
===============

data/
  Contains data files:

  bold.nii.gz
    The motion-corrected 4D timeseries (1452 volumes with 40 x 64 x 64 voxels,
    corresponding to a voxel size of 3.5 x 3.75 x 3.75 mm and a volume repetition
    time of 2.5 seconds). The timeseries contains all 12 runs of the original
    experiment, concatenated in a single file. Please note, that the timeseries
    signal is *not* detrended.

  bold_mc.par
    The motion correction parameters. This is a 6-column textfile with
    three rotation and three translation parameters respectively. This
    information can be used e.g. as additional regressors for :ref:`motion-aware
    timeseries detrending <motion-aware_detrending>`.

  mask*.nii.gz
    A number of mask images in the subjects functional space, including a
    full-brain mask.

  attributes.txt
    A two-column text file with the stimulation condition and the corresponding
    experimental run for each volume in the timeseries image. The labels are given
    in literal form (e.g. 'face').

  anat.nii.gz
    An anatomical image of the subject, projected and resampled into the same
    space as the functional images, hence also of the same spatial resolution. The
    image is *not* skull-stripped.

results/
  Some analyses presented in the tutorial takes non-negligible time to
  compute. Therefore, we provide results of some analysis so they
  could simply be loaded while following the tutorial (commands to
  load them are embedded in the code snippets through out tutorial and
  prefixed with ``# alt: ``).

start_tutorial_session.sh
  Helper shell script to start an interactive session within IPython
  to proceed with the tutorial code.

tutorial_lib.py
  Helper Python module used through out the tutorial to avoid
  presenting sequences of common operations (e.g. loading data)
  multiple times.

Instructions
============

  >>> from mvpa.suite import *
  >>> datapath = os.path.join(pymvpa_datadbroot, 'tutorial_data',
  ...                         'tutorial_data', 'data')
  >>> attrs = SampleAttributes(os.path.join(datapath, 'attributes.txt'))
  >>> ds = fmri_dataset(samples=os.path.join(datapath, 'bold.nii.gz'),
  ...                   targets=attrs.targets, chunks=attrs.chunks,
  ...                   mask=os.path.join(datapath, 'mask_brain.nii.gz'))
  >>> print ds.shape
  (1452, 39912)
  >>> print ds.a.voxel_dim
  (40, 64, 64)
  >>> print ds.a.voxel_eldim
  (3.5, 3.75, 3.75)
  >>> print ds.a.mapper
  <Chain: <Flatten>-<StaticFeatureSelection>>
  >>> print ds.uniquetargets
  ['bottle' 'cat' 'chair' 'face' 'house' 'rest' 'scissors' 'scrambledpix'
   'shoe']


References
==========

:ref:`Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J., and Pietrini,
pl.  (2001) <HGF+01>`. Distributed and overlapping representations of faces and
objects in ventral temporal cortex. Science 293, 2425–2430.


.. _FSL: http://www.fmrib.ox.ac.uk/fsl


Changelog
=========

0.2

  * Updated tutorial code to work with PyMVPA 0.6
  * Removed dependency on PyNIfTI and use NiBabel instead.

0.1

  * Initial release.
