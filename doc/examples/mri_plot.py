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
Basic (f)MRI plotting
=====================

.. index:: plotting

Estimate basic univariate sensitivity (ANOVA) an plot it overlayed on top
of the anatomical.

We start with basic steps: loading PyMVPA and the example fMRI
dataset, basic preprocessing, estimation of the ANOVA scores and
plotting.
"""

from mvpa.suite import *

# load PyMVPA example dataset
datapath = os.path.join(pymvpa_dataroot, 'demo_blockfmri', 'demo_blockfmri')
attr = SampleAttributes(os.path.join(datapath, 'attributes.txt'))
dataset = fmri_dataset(samples=os.path.join(datapath, 'bold.nii.gz'),
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask=os.path.join(datapath, 'mask_brain.nii.gz'))

# do chunkswise linear detrending on dataset
poly_detrend(dataset, chunks='chunks')

# define sensitivity analyzer
sensana = OneWayAnova(mapper=absolute_features())
sens = sensana(dataset)

"""
It might be convenient to pre-define common arguments for multiple calls to
plot_lightbox
"""

mri_args = {
    'background' : os.path.join(datapath, 'anat.nii.gz'),
    'background_mask' : os.path.join(datapath, 'mask_brain.nii.gz'),
    'overlay_mask' : os.path.join(datapath, 'mask_brain.nii.gz'),
    'do_stretch_colors' : False,
    'cmap_bg' : 'gray',
    'cmap_overlay' : 'autumn', # YlOrRd_r # P.cm.autumn
    'fig' : None,              # create new figure
    'interactive' : cfg.getboolean('examples', 'interactive', True),
    }

fig = plot_lightbox(overlay=dataset.map2nifti(sens),
              vlim=(0.5, None),
              #vlim_type="symneg_z",
              **mri_args)


"""
Output of the example analysis:

.. image:: ../pics/ex_plot_lightbox.*
   :align: center
   :alt: Simple plotting facility for (f)MRI

"""
