#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
==========================================
 Using scikit-learn regressors with PyMVPA
==========================================

Scikit-learn is a rich library of algorithms, many of them implementing the
`estimator and predictor API`_. PyMVPA provides the wrapper class,
:class:`~mvpa2.clfs.skl.base.SKLLearnerAdapter` that enables the use
of all of these algorithms within the PyMVPA framework. With this adaptor
these aspects of the scikit-learn API are presented through a PyMVPA
learner interface that is fully compatible with all other building blocks of
PyMVPA.

In this example we demonstrate this interface by mimicking the "`Decision 
Tree Regression`_" example from the scikit-learn documentation --
applying the minimal modifications necessary to the scikit-learn decision tree 
regression algorithm (with two different paramter settings) implementation on 
a PyMVPA dataset.

.. _estimator and predictor API: http://scikit-learn.org/stable/developers/#apis-of-scikit-learn-objects
.. _Decision Tree Regression: http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html 
"""


print(__doc__)

import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

"""
So far the code has been identical. The first difference is the import of the
adaptor class. We also use a convenient way to convert the data into a proper
:class:`~mvpa2.datasets.base.Dataset`.
"""

# this first import is only required to run the example a part of the test suite
from mvpa2 import cfg
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from mvpa2.suite import *
ds_train=dataset_wizard(samples=X, targets=y)


"""
The following lines are an example of the only significant modification
with respect to a pure scikit-learn implementation: the regressor is
wrapped into the adaptor. The result is a PyMVPA classifier, hence can 
be called with a dataset that contains both samples and targets.
"""

clf_1 = SKLLearnerAdapter(DecisionTreeRegressor(max_depth=2))
clf_2 = SKLLearnerAdapter(DecisionTreeRegressor(max_depth=5))

clf_1.train(ds_train)
clf_2.train(ds_train)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = clf_1.predict(X_test)
y_2 = clf_2.predict(X_test)


# plot the results
# which clearly show the overfitting for the second depth setting  
import pylab as pl

pl.figure()
pl.scatter(X, y, c="k", label="data")
pl.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
pl.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Decision Tree Regression")
pl.legend()


"""
This example shows that a PyMVPA classifier can be used in pretty much the
same way as the corresponding scikit-learn API. What this example does not show
is that with the :class:`~mvpa2.clfs.skl.base.SKLLearnerAdapter` class any
scikit-learn classifier can be employed in arbitrarily complex PyMVPA
processing pipelines and is enhanced with automatic training and all other
functionality of PyMVPA classifier implementations.
"""

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()