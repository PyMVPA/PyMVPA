#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Using scikit-learn Transformers in PyMVPA 
=========================================

PyMVPA provides a wrapper class, namely SKLTransformer, that allows the use of
an arbitrary scikit-learn transformer as a mapper. Here we demonstate its use 
by showing how nonlinear dimensionality reduction (manifold learning) can be 
performed with Multidimensional Scaling (MDS) as provided by scikit-learn.

This script was tested with version 0.14.1 of scikit-learn.
"""


# we use the S-curve dataset from scikit-learn
# with 1000 sample points
from mvpa2.datasets.sources.sklearn_data import skl_s_curve
ds = skl_s_curve(1000)

# shortcuts to resemble scikit-learn nomenclature
X = ds.samples
color = ds.targets

# create an instance of MDS from scikit-learn
# and wrap it by SKLTransformer
from sklearn.manifold import MDS
from mvpa2.mappers.skl_adaptor import SKLTransformer
mds = SKLTransformer(MDS(n_components=2, max_iter=100, n_init=1)) 

# call mds with the similarities of the input data based on Euclidean distances   
# (see scikit-learn documentation for details on the algorithm)
# SKLTransformer instances do require a dataset as input  
# TODO: find a better way to do this in PyMVPA
from sklearn.metrics import euclidean_distances
from mvpa2.datasets.base import Dataset
Y = mds(Dataset(euclidean_distances(X)))


# plotting of the result for illustration purposes
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

fig = pl.figure(figsize=(15, 8))
pl.suptitle("Manifold Learning using Multidimensional Scaling (MDS)"
           +" from scikit-learn" , fontsize=14)

# plot original 3D data
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=pl.cm.Spectral)
ax.view_init(4, -72)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())

# plot MDS result, that is the data reduction to 2D
ax = fig.add_subplot(122)
ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=pl.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

pl.show()
