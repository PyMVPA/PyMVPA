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
Visualization of Data Projection Methods
========================================
"""

from mvpa.misc.data_generators import noisy_2d_fx
from mvpa.mappers.svd import SVDMapper
from mvpa.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa import cfg

import pylab as P
import numpy as N
center = [10, 20]
axis_range = 7

##REF: Name was automagically refactored
def plot_proj_dir(p):
    P.plot([0, p[0,0]], [0, p[0,1]],
           linewidth=3, hold=True, color='y')
    P.plot([0, p[1,0]], [0, p[1,1]],
           linewidth=3, hold=True, color='k')

mappers = {
            'PCA': PCAMapper(),
            'SVD': SVDMapper(),
            'ICA': ICAMapper(alg='CuBICA'),
          }
datasets = [
    noisy_2d_fx(100, lambda x: x, [lambda x: x],
                center, noise_std=0.5),
    noisy_2d_fx(50, lambda x: x, [lambda x: x, lambda x: -x],
                center, noise_std=0.5),
    noisy_2d_fx(50, lambda x: x, [lambda x: x, lambda x: 0],
                center, noise_std=0.5),
   ]

ndatasets = len(datasets)
nmappers = len(mappers.keys())

P.figure(figsize=(8,8))
fig = 1

for ds in datasets:
    for mname, mapper in mappers.iteritems():
        mapper.train(ds)

        dproj = mapper.forward(ds.samples)
        mproj = mapper.proj
        P.subplot(ndatasets, nmappers, fig)
        if fig <= 3:
            P.title(mname)
        P.axis('equal')

        P.scatter(ds.samples[:, 0] - center[0],
                  ds.samples[:, 1] - center[1],
                  s=30, c=(ds.sa.targets) * 200)
        plot_proj_dir(mproj)
        fig += 1


if cfg.getboolean('examples', 'interactive', True):
    P.show()

"""
Output of the example:

.. image:: ../pics/ex_projections.*
   :align: center
   :alt: SVD/ICA/PCA projections

"""
