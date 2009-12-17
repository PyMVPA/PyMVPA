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
attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes_literal.txt'),
                        literallabels=True)
dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

# since we don't have a proper anatomical -- lets overlay on BOLD
nianat = NiftiImage(dataset.O[0], header=dataset.a.imghdr)

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, model='linear')

# define sensitivity analyzer
sensana = OneWayAnova(transformer=N.abs)
sens = sensana(dataset)

"""
It might be convinient to pre-define common arguments for multiple calls to
plotMRI
"""
mri_args = {
    'background' : nianat,              # could be a filename
    'background_mask' : os.path.join(pymvpa_dataroot, 'mask.nii.gz'),
    'overlay_mask' : os.path.join(pymvpa_dataroot, 'mask.nii.gz'),
    'do_stretch_colors' : False,
    'cmap_bg' : 'gray',
    'cmap_overlay' : 'autumn', # YlOrRd_r # P.cm.autumn
    'fig' : None,              # create new figure
    'interactive' : cfg.getboolean('examples', 'interactive', True),
    }

fig = plotMRI(overlay=dataset.map2nifti(sens),
              vlim=(0.5, None),
              #vlim_type="symneg_z",
              **mri_args)


"""
Output of the example analysis:

.. image:: ../pics/ex_plotMRI.*
   :align: center
   :alt: Simple plotting facility for (f)MRI

"""
