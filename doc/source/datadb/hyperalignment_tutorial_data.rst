.. _datadb_hyperalignment_tutorial_data:

**********************************************************************************
Hypearalignment Tutorial Data: Faces and Objects in Ventral Temporal Cortex (fMRI)
**********************************************************************************

This is a block-design fMRI dataset from a study on face and object
representation in human ventral temporal cortex.  It consists of 10 subjects
with 8 runs per subject. In each run, the subjects passively viewed greyscale
images from seven face & object categories. Each image was shown for 500ms
and was followed by a 1500ms inter-stimulus interval.  Full-brain fMRI data
were recorded with a volume repetition time of 2.5s, thus, a stimulus block
was covered by roughly 16 volumes. For a complete description of the experimental
design, fMRI acquisition parameters, and preprocessing steps, and previously
obtained results see the references_ below.

This tutorial dataset is based on data from a study published by :ref:`Haxby
et al. (2011) <HGC+11>`. The datasets have been preprocessed to a degree that should
allow people without prior fMRI experience to perform meaningful analyses.
Moreover, it should not require further preprocessing with external tools.

All preprocessing has been performed using tools from AFNI & PyMVPA.
Specifically, the 4D fMRI timeseries has been preprocessed as described in
:ref: `Haxby et al. (2011) <HGC+11>` and aligned to the standard MNI brain.
A Ventral Temporal Cortex mask in MNI space is applied to the data.

Terms Of Use
============

The original authors of :ref:`Haxby et al. (2011) <HGC+11>` hold the copyright
of this dataset and made it available under the terms of the `Creative Commons
Attribution-Share Alike 3.0`_ license.

.. _Creative Commons Attribution-Share Alike 3.0: http://creativecommons.org/licenses/by-sa/3.0/


Download
========

A single compressed hdf5 is available at:

  http://data.pymvpa.org/datasets/hyperalignment_tutorial_data


Content
=======================

hyperalignment_tutorial_data.hdf5.gz
    The list of datasets for 10 subjects stored as a compressed hdf5 file.
    Each dataset contains category & run labels.

hyperalignment_tutorial_data_2.4.hdf5.gz
    This file contains the same data as the first one, but it can be opened
    on systems that have no NiBabel installed. Loading this file requires
    PyMVPA version 2.4 or later.


Instructions
============

  >>> from mvpa2.suite import *
  >>> filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data',
  ...                         "hyperalignment_tutorial_data.hdf5.gz")
  >>> datasets = h5load(filepath)
  >>> print len(datasets)
  10
  >>> print datasets[0]
  <Dataset: 56x3509@float32, <sa: chunks,targets,time_coords,time_indices>, <fa: voxel_indices>, <a: imghdr,imgtype,mapper,voxel_dim,voxel_eldim>>


References
==========

:ref:`Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., Hanke, M. & Ramadge, P. J. (2011) <HGC+11>.`
A Common, High-Dimensional Model of the Representational Space in Human
Ventral Temporal Cortex. Neuron, 72, 404–416.
DOI: http://dx.doi.org/10.1016/j.neuron.2011.08.026
