The MNIST Database Of Handwritten Digits
----------------------------------------

.. note::

  This dataset is courtesy of Yann LeCun (Courant Institute, NYU) and
  Corinna Cortes (Google Labs, New York), who distribute it through their
  website at: http://yann.lecun.com/exdb/mnist/

  The version that is offered here is identical to the four files distributed
  there, but has been converted into a single HDF5 file than can easily be read
  by PyMVPA.

The MNIST database of handwritten digits, available from this page, has a
training set of 60,000 examples, and a test set of 10,000 examples. It is a
subset of a larger set available from NIST. The digits have been
size-normalized and centered in a fixed-size image.  It is a good database for
people who want to try learning techniques and pattern recognition methods on
real-world data while spending minimal efforts on preprocessing and formatting.

See http://yann.lecun.com/exdb/mnist for more information.


Requirements
------------

* HDF5 access utilizes the H5PY_ package.
* *PyMVPA 0.5* (or later) provides the `h5load()` function.

.. _H5PY: http://h5py.alfven.org/


Instructions
------------

  >>> from mvpa.suite import *
  >>> filepath = os.path.join(pymvpa_dataroot, 'mnist', "mnist.hdf5")
  >>> datasets = h5load(filepath)
  >>> train = datasets['train']
  >>> test = datasets['test']
  >>> print train
  <Dataset: 60000x784@uint8, <sa: labels>, <a: mapper>>
  >>> print test
  <Dataset: 10000x784@uint8, <sa: labels>, <a: mapper>>


References
----------

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning
applied to document recognition." Proceedings of the IEEE, 86: 2278-2324,
November 1998.
