.. _datadb_mnist:

*********************************************************************
LeCun et al. (1999): The MNIST Dataset Of Handwritten Digits (Images)
*********************************************************************

The MNIST_ dataset of handwritten digits, available from this page, has a
training set of 60,000 examples, and a test set of 10,000 examples. It is a
subset of a larger set available from NIST.  The digits have been
size-normalized and centered in a fixed-size image.  It is a good database for
people who want to try learning techniques and pattern recognition methods on
real-world data while spending minimal efforts on preprocessing and formatting.

See http://yann.lecun.com/exdb/mnist for more information.

.. note::

  The version that is offered here is identical to the four files distributed
  there, but has been converted into a single HDF5 file than can easily be read
  by PyMVPA.


Terms Of Use
============

`Yann LeCun`_ (Courant Institute, NYU) and `Corinna Cortes`_ (Google
Labs, New York) hold the copyright of MNIST_ dataset, which is a
derivative work from original NIST datasets.  MNIST_ dataset is made
available under the terms of the `Creative Commons Attribution-Share
Alike 3.0`_ license.

.. _MNIST: http://yann.lecun.com/exdb/mnist
.. _Creative Commons Attribution-Share Alike 3.0: http://creativecommons.org/licenses/by-sa/3.0/
.. _Yann LeCun: http://yann.lecun.com/
.. _Corinna Cortes: http://web.me.com/corinnacortes/work/Home.html


Download
========

A single hdf5 file containing entire MNIST_ dataset is available from

  http://data.pymvpa.org/datasets/mnist/


Requirements
============

* HDF5 access facility.
* *PyMVPA 0.5* (or later) provides the `h5load()` function (utilizes H5PY_ package).

.. _H5PY: http://h5py.alfven.org/


Instructions
============

  >>> from mvpa2.suite import *
  >>> filepath = os.path.join(pymvpa_datadbroot, 'mnist', "mnist.hdf5")
  >>> datasets = h5load(filepath)
  >>> train = datasets['train']
  >>> test = datasets['test']
  >>> print train
  <Dataset: 60000x784@uint8, <sa: labels>>
  >>> print test
  <Dataset: 10000x784@uint8, <sa: labels>>
  >>> # assign a mapper able to recreate 28x28 pixel image arrays
  >>> test.a.mapper = FlattenMapper(shape=(28, 28))
  >>> test.mapper.reverse(test).shape
  (10000, 28, 28)


References
==========

:ref:`LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998) <LBB+98>`.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE, 86, 2278--2324.
