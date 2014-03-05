#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
============================================
 Using scikit-learn classifiers with PyMVPA
============================================

Scikit-learn is a rich library of algorithms, many of them implementing the
`estimator and predictor API`_. PyMVPA provides the wrapper class,
:class:`~mvpa2.clfs.skl.base.SKLLearnerAdapter` that enables the use
of all of these algorithms within the PyMVPA framework. With this adaptor
these aspects of the scikit-learn API are presented through a PyMVPA
learner interface that is fully compatible with all other building blocks of
PyMVPA.

In this example we demonstrate this interface by mimicking the "`Nearest 
Neighbors Classification`_" example from the scikit-learn documentation --
applying the minimal modifications necessary to run two variants of the 
scikit-learn k-nearest neighbors algorithm implementation on PyMVPA datasets.

.. _estimator and predictor API: http://scikit-learn.org/stable/developers/#apis-of-scikit-learn-objects
.. _Nearest Neighbors Classification: http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
"""

print(__doc__)

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors

n_neighbors = 15

"""
So far the code has been identical. The first difference is the import of the
adaptor class. We also load the scikit-learn demo dataset, but also with the
help of a wrapper function that yields a PyMVPA dataset.
"""


# this first import is only required to run the example a part of the test suite
from mvpa2 import cfg
from mvpa2.clfs.skl.base import SKLLearnerAdapter

# load the iris dataset
from mvpa2.datasets.sources.skl_data import skl_iris
iris = skl_iris()
# compact dataset summary
print iris

"""
The original example uses only the first two features of the dataset, 
since it intends to visualize learned classification boundaries in 2-D.
We can do the same slicing directly on our PyMVPA dataset.
"""

iris=iris[:,[0,1]]

d = {'setosa':0, 'versicolor':1, 'virginica':2}

"""
For visualization we will later map the literal class labels onto
numerival values. Besides that, we continue with practically identical code.
"""

h = .02  # step size in the mesh

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # create an instance of the algorithm from scikit-learn,
    # wrap it by SKLLearnerAdapter and finally train it
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    wrapped_clf=SKLLearnerAdapter(clf)
    wrapped_clf.train(iris)

    """
    The following lines are an example of the only significant modification
    with respect to a pure scikit-learn implementation: the classifier is
    wrapped into the adaptor.  The result is a PyMVPA classifier, hence can 
    be called with a dataset that contains both samples and targets.
    """

    # shortcut
    X = iris.samples
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = wrapped_clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # to put the result into a color plot we now need numerical values
    # this can be done nicely in PyMVPA
    from mvpa2.datasets.formats import AttributeMap
    Z = AttributeMap().to_numeric(Z)
    Z = Z.reshape(xx.shape)

    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # For plotting the training points we convert to numerical values again
    pl.scatter(X[:, 0], X[:, 1], c=AttributeMap().to_numeric(iris.targets), 
                                 cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title("3-Class classification (k = %i, weights = '%s')"
             % (n_neighbors, weights))

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()

"""
This example shows that a PyMVPA classifier can be used in pretty much the
same way as the corresponding scikit-learn API. What this example does not show
is that with the :class:`~mvpa2.clfs.skl.base.SKLLearnerAdapter` class any
scikit-learn classifier can be employed in arbitrarily complex PyMVPA
processing pipelines and is enhanced with automatic training and all other
functionality of PyMVPA classifier implementations.
"""
