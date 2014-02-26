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
Representational similarity analysis (RSA) on fMRI data
=======================================================

.. index:: rsa_fmri

"""

import numpy as np
import pylab as pl
from mvpa2 import cfg

# load dataset -- ventral and occipital ROIs
from mvpa2.misc.data_generators import load_datadb_tutorial_data
ds = load_datadb_tutorial_data(roi=(15, 16, 23, 24, 36, 38, 39, 40, 48))

# only minial detrending
from mvpa2.mappers.detrend import poly_detrend
poly_detrend(ds, polyord=1, chunks_attr='chunks')
# z-scoring with respect to the 'rest' condition
from mvpa2.mappers.zscore import zscore
zscore(ds, chunks_attr='chunks', param_est=('targets', 'rest'))
# now remove 'rest' samples
ds = ds[ds.sa.targets != 'rest']

# little helper function to plot dissimilarity matrices
def plot_mtx(mtx, labels, title):
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0,1))
    pl.colorbar()

# compute a dataset with the mean samples for all conditions
from mvpa2.mappers.fx import mean_group_sample
mtgs = mean_group_sample(['targets'])
mtds = mtgs(ds)

# basic ROI RSA -- dissimilarity matrix for the entire ROI
from mvpa2.measures import rsa
dsm = rsa.PDist(square=True)
res = dsm(mtds)
plot_mtx(res, mtds.sa.targets, 'ROI pattern correlation distances')

# same as above, but done in a searchlight fashion
from mvpa2.measures.searchlight import sphere_searchlight
dsm = rsa.PDist(square=False)
sl = sphere_searchlight(dsm, 2)
slres = sl(mtds)
# score each searchlight sphere result wrt global pattern dissimilarity
distinctiveness = np.sum(np.abs(slres), axis=0)
print 'Most dissimilar patterns around', \
        mtds.fa.voxel_indices[distinctiveness.argmax()]
# take a look at the this dissimilarity structure
from scipy.spatial.distance import squareform
plot_mtx(squareform(slres.samples[:, distinctiveness.argmax()]),
         mtds.sa.targets,
         'Maximum distinctive searchlight pattern correlation distances')

# more interesting: let's look at the stability of similarity sturctures
# across experiment runs
# mean condition samples, as before, but now individually for each run
mtcgs = mean_group_sample(['targets', 'chunks'])
mtcds = mtcgs(ds)

# searchlight consistency measure -- how correlated are the structures
# across runs
dscm = rsa.PDistConsistency()
sl_cons = sphere_searchlight(dscm, 2)
slres_cons = sl_cons(mtcds)
# mean correlation
mean_consistency = np.mean(slres_cons, axis=0)
print 'Most stable dissimilarity patterns around', \
        mtds.fa.voxel_indices[mean_consistency.argmax()]
# Look at this pattern
plot_mtx(squareform(slres.samples[:, mean_consistency.argmax()]),
         mtds.sa.targets,
         'Most consistent searchlight pattern correlation distances')

# let's see where in the brain we find dissimilarity structures that are
# similar to out most stable one
tdsm = rsa.PDist2Target(
            slres.samples[:, mean_consistency.argmax()])
# using a searchlight
from mvpa2.base.learner import ChainLearner
from mvpa2.mappers.shape import TransposeMapper
sl_tdsm = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), 2)
slres_tdsm = sl_tdsm(mtds)

# plot the spatial distribution using NiPy
vol = ds.a.mapper.reverse1(slres_tdsm.samples[0])
import nibabel as nb
anat = nb.load('datadb/tutorial_data/tutorial_data/data/anat.nii.gz')

from nipy.labs.viz_tools.activation_maps import plot_map
pl.figure(figsize=(15,4))
sp = pl.subplot(121)
pl.title('Distribution of target similarity structure correlation')
slices = plot_map(
            vol,
            ds.a.imghdr.get_best_affine(),
             cut_coords=np.array((12,-42,-20)),
             threshold=.5,
             cmap="bwr",
             vmin=0,
             vmax=1.,
             axes=sp,
             anat=anat.get_data(),
             anat_affine=anat.get_affine(),
         )
img = pl.gca().get_images()[1]
cax = pl.axes([.05, .05, .05, .9])
pl.colorbar(img, cax=cax)

sp = pl.subplot(122)
pl.hist(slres_tdsm.samples[0],
        #range=(0,410),
        normed=False,
        bins=30,
        color='0.6')
pl.ylabel("Number of voxels")
pl.xlabel("Target similarity structure correlation")



if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()
