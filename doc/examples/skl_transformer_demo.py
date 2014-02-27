#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=============================================
 Using scikit-learn transformers with PyMVPA
=============================================

Scikit-learn is a rich library of algorithms, many of them implementing the
`transformer API`_. PyMVPA provides a wrapper class,
:class:`~mvpa2.mappers.skl_adaptor.SKLTransformer` that enables the use
of all of these algorithms within the PyMVPA framework. With this adaptor
the transformer API is presented as a PyMVPA mapper interface that is fully
compatible with all other building blocks of PyMVPA.

In this example we demonstrate this interface by mimicking the "`Comparison of
Manifold Learning methods`_" example from the scikit-learn documentation --
applying the minimal modifications necessary to run a variety of scikit-learn
algorithm implementation on PyMVPA datasets.

This script also prints the same timing information as the original.

.. _transformer API: http://scikit-learn.org/stable/developers/#apis-of-scikit-learn-objects
.. _Comparison of Manifold Learning methods: http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html

"""

print(__doc__)

from time import time

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
# Next line to silence pyflakes. This import is needed.
Axes3D

n_points = 1000
n_neighbors = 10
n_components = 2

"""
So far the code has been identical. The first difference is the import of the
adaptor class. We also load the scikit-learn demo dataset, but also with the
help of a wrapper function that yields a PyMVPA dataset.
"""

# this first import is only required to run the example a part of the test suite
from mvpa2 import cfg
from mvpa2.mappers.skl_adaptor import SKLTransformer

# load the S-curve dataset 
from mvpa2.datasets.sources.skl_data import skl_s_curve
ds = skl_s_curve(n_points)

"""
And we continue with practically identical code.
"""

fig = pl.figure(figsize=(15, 8))
pl.suptitle("Manifold Learning with %i points, %i neighbors"
            % (1000, n_neighbors), fontsize=14)

try:
    # compatibility matplotlib < 1.0
    X = ds.samples
    ax = fig.add_subplot(241, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=ds.targets, cmap=pl.cm.Spectral)
    ax.view_init(4, -72)
except:
    X = ds.samples
    ax = fig.add_subplot(241, projection='3d')
    pl.scatter(X[:, 0], X[:, 2], c=ds.targets, cmap=pl.cm.Spectral)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    # create an instance of the algorithm from scikit-learn
    # and wrap it by SKLTransformer

    """
    The following lines are an example of the only significant modification
    with respect to a pure scikit-learn implementation: the transformer is
    wrapped into the adaptor.  The result is a mapper, hence can be called
    with a dataset that contains both samples and targets -- without explcitly
    calling ``fit()`` and ``transform()``.
    """

    lle = SKLTransformer(manifold.LocallyLinearEmbedding(n_neighbors, 
                                                         n_components,
                                                         eigen_solver='auto',
                                                         method=method))
    # call the SKLTransformer instance on the input dataset
    Y = lle(ds)

    """
    The rest of the example is unmodified except for the wrapping of the
    respective transformer into the Mapper adaptor.
    """

    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(242 + i)
    pl.scatter(Y[:, 0], Y[:, 1], c=ds.targets, cmap=pl.cm.Spectral)
    pl.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    pl.axis('tight')

t0 = time()
# create an instance of the algorithm from scikit-learn
# and wrap it by SKLTransformer
iso = SKLTransformer(manifold.Isomap(n_neighbors=10, n_components=2))
# call the SKLTransformer instance on the input dataset
Y = iso(ds)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(246)
pl.scatter(Y[:, 0], Y[:, 1], c=ds.targets, cmap=pl.cm.Spectral)
pl.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')


t0 = time()
# create an instance of the algorithm from scikit-learn
# and wrap it by SKLTransformer
mds = SKLTransformer(manifold.MDS(n_components=2, max_iter=100, 
                                  n_init=1, dissimilarity='euclidean')) 
# call the SKLTransformer instance on the input dataset
Y = mds(ds)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(247)
pl.scatter(Y[:, 0], Y[:, 1], c=ds.targets, cmap=pl.cm.Spectral)
pl.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')


t0 = time()
# create an instance of the algorithm from scikit-learn
# and wrap it by SKLTransformer
se = SKLTransformer(manifold.SpectralEmbedding(n_components=n_components,
                                               n_neighbors=n_neighbors))
# call the SKLTransformer instance on the input dataset
Y = se(ds)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(248)
pl.scatter(Y[:, 0], Y[:, 1], c=ds.targets, cmap=pl.cm.Spectral)
pl.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')



if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()
