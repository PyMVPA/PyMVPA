"""
=============================================================
 Comparison of Manifold Learning methods using SKLTransformer
=============================================================

PyMVPA provides a wrapper class, namely SKLTransformer, that allows the use of
an arbitrary scikit-learn transformer as a mapper. Here we demonstate its 
simple application by cloning the scikit-learn example for various manifold 
learning methods (dimensionality reduction) on the S-curve dataset  
(see scikit_learn_.).

.. _scikit_learn: http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html

"""



print(__doc__)

from time import time

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from mvpa2.mappers.skl_adaptor import SKLTransformer

# Next line to silence pyflakes. This import is needed.
Axes3D

# load the S-curve dataset 
from mvpa2.datasets.sources.sklearn_data import skl_s_curve
n_points = 1000
ds = skl_s_curve(n_points)
# shortcuts to resemble scikit-learn nomenclature
X = ds.samples
color = ds.targets

n_neighbors = 10
n_components = 2

fig = pl.figure(figsize=(15, 8))
pl.suptitle("Manifold Learning with %i points, %i neighbors"
            % (1000, n_neighbors), fontsize=14)

try:
    # compatibility matplotlib < 1.0
    ax = fig.add_subplot(241, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=pl.cm.Spectral)
    ax.view_init(4, -72)
except:
    ax = fig.add_subplot(241, projection='3d')
    pl.scatter(X[:, 0], X[:, 2], c=color, cmap=pl.cm.Spectral)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    # create an instance of the algorithm from scikit-learn
    # and wrap it by SKLTransformer
    lle = SKLTransformer(manifold.LocallyLinearEmbedding(n_neighbors, 
                                                         n_components,
                                                         eigen_solver='auto',
                                                         method=method))
    # call the SKLTransformer instance on the input dataset
    Y = lle(ds)                                                     
    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(242 + i)
    pl.scatter(Y[:, 0], Y[:, 1], c=color, cmap=pl.cm.Spectral)
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
pl.scatter(Y[:, 0], Y[:, 1], c=color, cmap=pl.cm.Spectral)
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
pl.scatter(Y[:, 0], Y[:, 1], c=color, cmap=pl.cm.Spectral)
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
pl.scatter(Y[:, 0], Y[:, 1], c=color, cmap=pl.cm.Spectral)
pl.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')

pl.show()
