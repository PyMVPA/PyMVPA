.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. index:: dataset, sample attribute, dataset attribute
.. _chap_datasets:

********
Datasets
********

The first step of any analysis in PyMVPA involves reading the data and putting
it into the necessary shape for the intended analysis. But even after the
initial setup, many algorithms have to modify datasets, e.g. by selecting a
subset of it, or simple transformations of the data (e.g. z-scoring), or more
complex things like projections into alternative representations/spaces.

This section introduces the basic concepts of a dataset in PyMVPA and shows
useful operations typically performed on datasets.


The Basic Concepts
==================

A minimal dataset in PyMVPA consists of a number of :term:`sample`\ s, where
each individual sample is nothing more than a vector of values. Each sample is
associated with a :term:`label`, which defines the category the respective
sample belongs to, or in more general terms, defines the model that should be
learned by a classifier.  Moreover, samples can be grouped into so-called
:term:`chunk`\ s, where each chunk is assumed to be statistically independent
from all other data chunks.

The foundation of PyMVPA's data handling is the :class:`~mvpa.datasets.base.Dataset` class. Basically, this
class stores data samples, sample attributes and dataset attributes.  By
definition, sample attributes assign a value to each data sample (e.g. labels,
or chunks) and dataset attributes are additional information or functionality
that apply to the whole dataset.

Most likely the :class:`~mvpa.datasets.base.Dataset` class will not be used directly, but through one
of the derived classes. However, it is perfectly possible to use it directly.
In the simplest case a dataset can be constructed by specifying some
data samples and the corresponding class labels.

  >>> import numpy as N
  >>> from mvpa.datasets import Dataset
  >>> data = Dataset(samples=N.random.normal(size=(10,5)), labels=1)
  >>> data
  <Dataset / float64 10 x 5 uniq: 1 labels 10 chunks>

.. index:: chunks, labels, feature, sample

The above example creates a dataset with 10 samples and 5 features each. The
values of all features stem from normally distributed random noise. The class
label '1' is assigned to all samples. Instead of a single scalar value `labels`
can also be a sequence with individual labels for each data sample. In this
case the length of this sequence has to match the number of samples.

Interestingly, the dataset object tells us about 10 `chunks`. In PyMVPA chunks
are used to group subsets of data samples. However, if no grouping information
is provided all data samples are assumed to be in their own group, hence no
sample grouping is performed.

Both `labels` and `chunks` are so called *sample attributes*. All sample
attributes are stored in sequence-type containers consisting of one value per
sample. These containers can be accessed by properties with the same as the
attribute:

  >>> data.labels
  array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  >>> data.chunks
  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

The *data samples* themselves are stored as a two-dimensional matrix
where each row vector is a `sample` and each column vector contains
the values of a `feature` across all `samples`. The :class:`~mvpa.datasets.base.Dataset` class
provides access to the samples matrix via the `samples` property.

  >>> data.samples.shape
  (10, 5)

The :class:`~mvpa.datasets.base.Dataset` class itself can only deal with 2d sample matrices. However,
PyMVPA provides a very easy way to deal with data where each data sample is
more than a 1d vector: `Data Mapping`_


.. index:: mapper, sample, feature

Data Mapping
============

It was already mentioned that the :class:`~mvpa.datasets.base.Dataset` class cannot deal with data samples
that are more than simple vectors. This could be a problem in cases where the
data has a higher dimensionality, e.g. functional brain-imaging data where
each data sample is typically a three-dimensional volume.

One approach to deal with this situation would be to concatenate the whole
volume into a 1d vector. While this would work in certain cases there is
definitely information lost. Especially for brain-imaging data one would most
likely want keep information about neighborhood and distances between data
sample elements.

In PyMVPA this is done by mappers that transform data samples from their
original *dataspace* into the so-called *features space*. In the above
neuro-imaging example the *dataspace* is three-dimensional and the *feature
space* always refers to the 2d `samples x features` representation that is
required by the :class:`~mvpa.datasets.base.Dataset` class. In the context of mappers the dataspace is
sometimes also referred to as *in-space* (i.e. the initial data that goes into
the mapper) while the feature space is labeled as *out-space* (i.e. the mapper
output when doing forward mapping).

The task of a mapper, besides transforming samples into 1d vectors, is to
retain as much information of the dataspace as possible. Some mappers provide
information about dataspace metrics and feature neighbourhood, but all mappers
are able to do reverse mapping from feature space into the original dataspace.

Usually one does not have to deal with mappers directly. PyMVPA provides some
convenience subclasses of :class:`~mvpa.datasets.base.Dataset` that automatically perform the necessary
mapping operations internally.

.. index:: MaskedDataset

For an introduction into to concept of a dataset with mapping capabilities
we can take a look at the :class:`~mvpa.datasets.masked.MaskedDataset` class. This dataset class works
almost exactly like the basic :class:`~mvpa.datasets.base.Dataset` class, except that it provides some
additional methods and is more flexible with respect to the format of the
sample data. A masked dataset can be created just like a normal dataset.

  >>> from mvpa.datasets.masked import MaskedDataset
  >>> mdata = MaskedDataset(samples=N.random.normal(size=(5,3,4)),
  ...                       labels=[1,2,3,4,5])
  >>> mdata
  <Dataset / float64 5 x 12 uniq: 5 chunks 5 labels>

However, unlike :class:`~mvpa.datasets.base.Dataset` the :class:`~mvpa.datasets.masked.MaskedDataset` class can deal with sample
data arrays with more than two dimensions. More precisely it handles arrays of
any dimensionality. The only assumption that is made is that the first axis
of a sample array separates the sample data points. In the above example we
therefore have 5 samples, where each sample is a 3x4 plane.

.. index:: forward mapping, reverse mapping

If we look at the self-description of the created dataset we can see that it
doesn't tell us about 3x4 plane, but simply 12 features. That is because
internally the sample array is automatically reshaped into the aforementioned
2d matrix representation of the :class:`~mvpa.datasets.base.Dataset` class. However, the information about
the original dataspace is not lost, but kept inside the mapper used by
:class:`~mvpa.datasets.masked.MaskedDataset`. Two useful methods of :class:`~mvpa.datasets.masked.MaskedDataset` make use of the mapper:
`mapForward()` and `mapReverse()`. The former can be used to transform
additional data from dataspace into the feature space and the latter performs
the same in the opposite direction.

  >>> mdata.mapForward(N.arange(12).reshape(3,4)).shape
  (12,)
  >>> mdata.mapReverse(N.array([1]*mdata.nfeatures)).shape
  (3, 4)

Especially reverse mapping can be very useful when visualizing classification
results and information maps on the original dataspace.

Another feature of mapped datasets is that valid mapping information is
maintained even when the feature space changes. When running some feature
selection algorithm (see :ref:`chap_featsel`) some features of the original features
set will be removed, but after feature selection one will most likely want
to know where the selected (or removed) features are in the original dataspace.
To make use of the neuro-imaging example again: The most convenient way to
access this kind of information would be a map of the selected features that
can be overlayed over some anatomical image. This is trivial with PyMVPA,
because the mapping is automatically updated upon feature selection.

  >>> mdata.mapReverse(N.arange(1,mdata.nfeatures+1))
  array([[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]])
  >>> sdata = mdata.selectFeatures([2,7,9,10])
  >>> sdata
  <Dataset / float64 5 x 4 uniq: 5 chunks 5 labels>
  >>> sdata.mapReverse(N.arange(1,sdata.nfeatures+1))
  array([[0, 0, 1, 0],
         [0, 0, 0, 2],
         [0, 3, 4, 0]])

.. index:: feature selection

The above example selects four features from the set of the 12 original
ones, by passing their ids to the `selectFeatures()` method. The method
returns a new dataset only containing the four selected features. Both datasets
share the sample data (using a NumPy array view). Using `selectFeatures()`
is therefore both memory efficient and relatively fast. All other
information like class labels and chunks are maintained. By calling
`mapReverse()` on the new dataset one can see that the remaining four features
are precisely mapped back onto their original locations in the data space.

.. index:: syntactic sugaring
.. _data_sugaring:

Data Access Sugaring
====================

Complementary to self-descriptive attribute names (e.g. `labels`, `samples`)
datasets have a few concise shortcuts to get quick access to some attributes
or perform some common action

================ ============ ================
Attribute        Abbreviation Definition class
---------------- ------------ ----------------
samples          S            :class:`~mvpa.datasets.base.Dataset`
labels           L            :class:`~mvpa.datasets.base.Dataset`
uniquelabels     UL           :class:`~mvpa.datasets.base.Dataset`
chunks           C            :class:`~mvpa.datasets.base.Dataset`
uniquechunks     UC           :class:`~mvpa.datasets.base.Dataset`
origids          I            :class:`~mvpa.datasets.base.Dataset`
samples_original O            :class:`~mvpa.datasets.mapped.MappedDataset`
================ ============ ================



.. index:: data formats
.. _data_formats:

Data Formats
============

The concept of mappers in conjunction with the functionality provided by the
:class:`~mvpa.datasets.base.Dataset` class, makes it very easy to create new dataset types with support
for specialized data types and formats. The following is a non-exhaustive list
of data formats currently supported by PyMVPA (for additional formats take a
look at the subclasses of :class:`~mvpa.datasets.base.Dataset`):

* NumPy arrays

  PyMVPA builds its dataset facilities on NumPy arrays. Basically, anything
  that can be converted into a NumPy array can also be converted into a
  dataset. Together with the corresponding labels, NumPy arrays can simply be
  passed to the :class:`~mvpa.datasets.base.Dataset` constructor to create a dataset. With arrays it is
  possible to use the classes :class:`~mvpa.datasets.base.Dataset`, :class:`~mvpa.datasets.mapped.MappedDataset` (to combine the samples
  with any custom mapping algorithm) or :class:`~mvpa.datasets.masked.MaskedDataset` (readily provides a
  :class:`~mvpa.mappers.array.DenseArrayMapper`).

* Plain text

  Using the NumPy function `fromfile()` a variety of text file formats (e.g.
  CSV) can be read and converted into NumPy arrays.

* NIfTI/Analyze images

  PyMVPA provides a specialized dataset for MRI data in the NIfTI format.
  :class:`~mvpa.datasets.nifti.NiftiDataset` uses PyNIfTI_ to read the data and automatically configures an
  appropriate :class:`~mvpa.mappers.array.DenseArrayMapper` with metric information read from the NIfTI
  file header.

* EEP binary files

  Another special dataset type is :class:`~mvpa.datasets.eep.EEPDataset`. It reads data from binary EEP
  file (written by eeprobe_)


.. _PyNIfTI: http://niftilib.sf.net/pynifti
.. _eeprobe: http://http://www.ant-neuro.com/products/eeprobe


.. index:: data splitting, splitter, leave-one-out
.. _data_splitter:

Data Splitting
==============

In many cases some algorithm should not run on a complete dataset, but just
some parts of it. One well-known example is leave-one-out cross-validation,
where a dataset is typically split into a number of training and validation
datasets. A classifier is trained on the training set and its generalization
performance is tested using the validation set.

It is important to strictly separate training and validation datasets
as otherwise no valid statement can be made whether a classifier
really generated an appropriate model of the training data. Violating this
requirement spuriously elevates the classification performance, often termed
'peeking' in the literature. However, they provide no relevant
information because they are based on cheating or peeking and do not
describe signal similarities between training and validation datasets.

.. <gjd> this point about 'peeking' is a critical one and
   maybe deserves emphasis. i was just looking at how we deal
   with it in our documentation, and we need to improve ours too!

With the splitter classes derived from the base
:class:`~mvpa.datasets.splitters.Splitter`,
PyMVPA makes dataset splitting easy. All dataset
splitters in PyMVPA are implemented as Python generators, meaning that when
called with a dataset once, they return one dataset split per iteration and
an appropriate Exception when they are done. This is exactly the same behavior
as of e.g. the Python `xrange()` function.

.. index:: working data, validation data

To perform data splitting for the already mentioned cross-validation, PyMVPA
provides the  :class:`~mvpa.datasets.splitters.NFoldSplitter` class. It implements a method to generate
arbitrary N-M splits, where N is the number of different chunks in a dataset
and M is any non-negative integer smaller than N. Doing a leave-one-out split
of our example dataset looks like this:

  >>> from mvpa.datasets.splitters import NFoldSplitter
  >>> splitter = NFoldSplitter(cvtype=1)   # Do N-1
  >>> for wdata, vdata in splitter(data):
  ...     pass

where `wdata` is the *working dataset* and `vdata` is the *validation dataset*.
If we have a look a those datasets we can see that the splitter did what we
intended:

  >>> split = [ i for i in splitter(data)][0]
  >>> for s in split:
  ...     print s
  Dataset / float64 9 x 5 uniq: 1 labels 9 chunks
  Dataset / float64 1 x 5 uniq: 1 labels 1 chunks
  >>> split[0].uniquechunks
  array([1, 2, 3, 4, 5, 6, 7, 8, 9])
  >>> split[1].uniquechunks
  array([0])

In the first split, the working dataset contains nine chunks of the original
dataset and the validation set contains the remaining chunk.

Behavior of the splitters can be heavily customized by additional arguments to
the constructor (see :class:`~mvpa.datasets.splitters.Splitter` for extended
help on the arguments).  For instance, in the analysis in fMRI data it might
be important to assure that samples in the training and testing parts of the
split are not neighboring samples (unless it is otherwise assured by the
presence of baseline condition on the boundaries between chunks, samples of
which are discarded prior the statistical learning analysis).  Providing
argument `discard_boundary=1` to the splitter, would remove from both training
and testing parts a single sample, which lie on the boundary between chunks.
Providing `discard_boundary=(2,0)` would remove 2 samples only from training
part of the split (which is desired strategy for `NFoldSplitter` where
training part contains majority of the data).

.. index:: processing object

The usage of the splitter, creating a splitter object and calling it with a
dataset, is a very common design pattern in the PyMVPA package. Like splitters,
there are many more so called *processing objects*. These classes or objects
are instantiated by passing all relevant parameters to the constructor.
Processing objects can then be called multiple times with different datasets
to perform their algorithm on the respective dataset. This design applies to
the majority of the algorithms implemented in PyMVPA.
