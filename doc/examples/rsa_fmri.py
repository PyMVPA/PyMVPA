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

In this example we are going to take a look at representational similarity
analysis (RSA). This term was coined by :ref:`Kriegeskorte et al. (2008)
<KMB08>` and refers to a technique were data samples are converted into a
self-referential distance space, in order to aid comparsion across domains. The
premise is that whenever no appropriate transformation is known to directly
compare two types of data directly (1st-level), it is still useful to compare
similarities computed in individual domains (2nd-level). For example, this
analysis technique has been used to identify inter-species commonalities in
brain response pattern variations during stimulation with visual objects
(single-cell recordings in monkeys compared to human fMRI, Krigeskorte et al.,
2008), and to relate brain response pattern similarities to predictions of
computational models of perception (Connolly et al., 2012).
"""

import numpy as np
import pylab as pl
from os.path import join as pjoin
from mvpa2 import cfg

"""
In this example we use a dataset from :ref:`Haxby et al. (2001) <HGF+01>` were
participants watched pictures of eight different visual objects, while fMRI was
recorded. The following snippet load a portion of this dataset (single subject)
from regions on the ventral and occipital surface of the brain.
"""

# load dataset -- ventral and occipital ROIs
from mvpa2.datasets.sources.native import load_tutorial_data
datapath = pjoin(cfg.get('location', 'tutorial data'), 'haxby2001')
ds = load_tutorial_data(roi=(15, 16, 23, 24, 36, 38, 39, 40, 48))

"""
We only do minimal pre-processing: linear trend removal and Z-scoring all voxel
time-series with respect to the mean and standard deviation of the "rest"
condition.
"""

# only minial detrending
from mvpa2.mappers.detrend import poly_detrend
poly_detrend(ds, polyord=1, chunks_attr='chunks')
# z-scoring with respect to the 'rest' condition
from mvpa2.mappers.zscore import zscore
zscore(ds, chunks_attr='chunks', param_est=('targets', 'rest'))
# now remove 'rest' samples
ds = ds[ds.sa.targets != 'rest']

"""
RSA is all about so-called dissimilarity matrices: square, symetric matrices
with a zero diagonal that encode the (dis)similarity between all pairs of
data samples or conditions in a dataset. We compose a little helper function
to plot such matrices, including a color-scale and proper labeling of matrix
rows and columns.
"""

# little helper function to plot dissimilarity matrices
def plot_mtx(mtx, labels, title):
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0,1))
    pl.colorbar()

"""
As a start, we want to inspect the dissimilarity structure of the stimulation
conditions in the entire ROI. For this purpose, we average all samples of
each conditions into a single examplar, using an FxMapper() instance.
"""

# compute a dataset with the mean samples for all conditions
from mvpa2.mappers.fx import mean_group_sample
mtgs = mean_group_sample(['targets'])
mtds = mtgs(ds)

"""
After these preparations we can use the PDist() measure to compute the desired
distance matrix -- by default using correlation distance as a metric. The
``square`` argument will cause a ful square matrix to be
produced, instead of a leaner upper-triangular matrix in vector form.
"""

# basic ROI RSA -- dissimilarity matrix for the entire ROI
from mvpa2.measures import rsa
dsm = rsa.PDist(square=True)
res = dsm(mtds)
plot_mtx(res, mtds.sa.targets, 'ROI pattern correlation distances')

"""
Inspecting the figure we can see that there is not much structure in the matrix,
except for the face and the house condition being slightly more dissimilar than
others.

Now, let's take a look at the variation of similarity structure through the
brain. We can plug the PDist() measure into a searchlight to quickly scan the
brain and harvest this information.
"""

# same as above, but done in a searchlight fashion
from mvpa2.measures.searchlight import sphere_searchlight
dsm = rsa.PDist(square=False)
sl = sphere_searchlight(dsm, 2)
slres = sl(mtds)

"""
The result is a compact distance matrix in vector form for each searchlight
location. We can now try to score each matrix. Let's find the distance matrix
with the largest overall distances across all stimulation conditions, i.e.
the location in the brain where brain response patterns are most dissimilar.
"""

# score each searchlight sphere result wrt global pattern dissimilarity
distinctiveness = np.sum(np.abs(slres), axis=0)
print 'Most dissimilar patterns around', \
        mtds.fa.voxel_indices[distinctiveness.argmax()]
# take a look at the this dissimilarity structure
from scipy.spatial.distance import squareform
plot_mtx(squareform(slres.samples[:, distinctiveness.argmax()]),
         mtds.sa.targets,
         'Maximum distinctive searchlight pattern correlation distances')

"""
That looks confusing. But how do we know that this is not just noise (it
probably is)? One way would be to look at how stable a distance matrix is,
when computed for different portions of a dataset.

To perform this analysis, we use another FxMapper() instance that averages
all data into a single sample per stimulation conditions, per ``chunk``. A
chunk in this context indicates a complete fMRI recording run.
"""

# more interesting: let's look at the stability of similarity sturctures
# across experiment runs
# mean condition samples, as before, but now individually for each run
mtcgs = mean_group_sample(['targets', 'chunks'])
mtcds = mtcgs(ds)

"""
With this dataset we can use PDistConsistency() to compute the similarity
of dissimilarity matrices computes from different chunks. And, of course,
it can be done in a searchlight.
"""

# searchlight consistency measure -- how correlated are the structures
# across runs
dscm = rsa.PDistConsistency()
sl_cons = sphere_searchlight(dscm, 2)
slres_cons = sl_cons(mtcds)

"""
Now we can determine the most brain location with the most stable
dissimilarity matrix.
"""

# mean correlation
mean_consistency = np.mean(slres_cons, axis=0)
print 'Most stable dissimilarity patterns around', \
        mtds.fa.voxel_indices[mean_consistency.argmax()]
# Look at this pattern
plot_mtx(squareform(slres.samples[:, mean_consistency.argmax()]),
         mtds.sa.targets,
         'Most consistent searchlight pattern correlation distances')

"""
It is all in the face!

It would be interesting to know where in the brain dissimilarity structures
can be found that are similar to this one. PDistTargetSimilarity() can
be used to discover this kind of information with any kind of target
dissimilarity structure. We need to transpose the result for aggregation
into a searchlight map, as PDistTargetSimilarity can return more features
than just the correlation coefficient.
"""

# let's see where in the brain we find dissimilarity structures that are
# similar to our most stable one
tdsm = rsa.PDistTargetSimilarity(
            slres.samples[:, mean_consistency.argmax()])
# using a searchlight
from mvpa2.base.learner import ChainLearner
from mvpa2.mappers.shape import TransposeMapper
sl_tdsm = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), 2)
slres_tdsm = sl_tdsm(mtds)

"""
Lastly, we can map this result back onto the 3D voxel grid, and overlay
it onto the brain anatomy.
"""

# plot the spatial distribution using NiPy
vol = ds.a.mapper.reverse1(slres_tdsm.samples[0])
import nibabel as nb
anat = nb.load(pjoin(datapath, 'sub001', 'anatomy', 'highres001.nii.gz'))

from nipy.labs.viz_tools.activation_maps import plot_map
pl.figure(figsize=(15,4))
sp = pl.subplot(121)
pl.title('Distribution of target similarity structure correlation')
slices = plot_map(
            vol,
            ds.a.imgaffine,
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
