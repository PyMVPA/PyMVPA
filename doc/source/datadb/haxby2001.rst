*****************************************************************
Haxby et al. (2001): Faces and Objects in Ventral Temporal Cortex
*****************************************************************

This is a block-design fMRI dataset from a study on face and object
representation in human ventral temporal cortex.  It consists of 6 subjects
with 12 runs per subject. In each run, the subjects passively viewed greyscale
images of eight object categories, grouped in 24s blocks separated by rest
periods. Each image was shown for 500ms and was followed by a 1500ms
inter-stimulus interval.  Full-brain fMRI data were recorded with a volume
repetition time of 2.5s, thus, a stimulus block was covered by roughly 9
volumes. This dataset has been repeatedly reanalyzed. For a complete
description of the experimental design, fMRI acquisition parameters, and
previously obtained results see the references_ below.


Terms Of Use
============

The orginal authors of :ref:`Haxby et al. (2001) <HGF+01>` hold the copyright
of this dataset and made it available under the terms of the `Creative Commons
Attribution-Share Alike 3.0`_ license.

.. _Creative Commons Attribution-Share Alike 3.0: http://creativecommons.org/licenses/by-sa/3.0/


Download
========

Separate tarballs for each subject are available at:

  http://data.pymvpa.org/db/haxby2001


Subject-tarball content
=======================

anat.nii.gz
    Highres anatomical image

bold.nii.gz
    4D fMRI timeseries image. (1452 volumes with 40 x 64 x 64 voxels,
    corresponding to a voxel size of 3.5 x 3.75 x 3.75 mm and a volume repetition
    time of 2.5 seconds). The timeseries contains all 12 runs of the original
    experiment, concatenated in a single file. Please note, that the timeseries
    signal is *not* detrended.

mask*.nii.gz
    Various masks in functional space provided by the original authors. `vt`
    refers to "ventral temporal", "face" and "house" masks are GLM contrast
    based localizer maps.

labels.txt
    A two-column text file with the stimulation condition and the corresponding
    experimental run for each volume in the timeseries image. The labels are
    given in literal form (e.g. 'face').


Instructions
============

  >>> from mvpa.suite import *
  >>> subjpath = os.path.join(pymvpa_dataroot, 'haxby2001', 'subj1')
  >>> attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'))
  >>> ds = fmri_dataset(samples=os.path.join(subjpath, 'bold.nii.gz'),
  ...                   labels=attrs.labels, chunks=attrs.chunks,
  ...                   mask=os.path.join(subjpath, 'mask.nii.gz'))
  >>> print ds


References
==========

:ref:`Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J., and Pietrini,
P.  (2001) <HGF+01>`. Distributed and overlapping representations of faces and
objects in ventral temporal cortex. Science 293, 2425–2430.

:ref:`Hanson, S., Matsuka, T., and Haxby, J. (2004) <HMH04>`. Combinatorial
codes in ventral temporal lobe for object recognition: Haxby (2001). revisited:
is there a “face” area? Neuroimage 23, 156–166.

:ref:`O’Toole, A. J., Jiang, F., Abdi, H., & Haxby, J. V. (2005) <OJA+05>`.
Partially distributed representations of objects and faces in ventral temporal
cortex.  Journal of Cognitive Neuroscience, 17, 580–590.

:ref:`Hanke, M., Halchenko, Y.O., Sederberg, P.B., Olivetti, E., Fründ, I.,
Rieger, J.W., Herrmann, C.S., Haxby, J.V., Hanson, S. and Pollmann, S (2009)
<HHS+09b>`. PyMVPA: a unifying approach to the analysis of neuroscientific
data. Frontiers in Neuroinformatics, 3:3.
