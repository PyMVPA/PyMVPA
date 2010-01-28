.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial3:

*************************************
Part 3: Mappers, The Swiss Army Knife
*************************************

In the :ref:`previous tutorial part <chap_tutorial2>` we have discovered a
magic ingredient of datasets: a mapper. Mappers are probably the most
powerful concept in PyMVPA, and there is little one would do without them.
As a matter of fact, even in the :ref:`first tutorial part
<chap_tutorial1>` we have used them already, without even seeing them.

In general, a mapper is an algorithm that transforms data of some kind.
This transformation can be as simple as selecting a subset of data, or as
complex as a multi-stage preprocessing pipeline. Some transformations are
reversible, others are not. Some are simple one-step computations, others
are iterative algorithms that have to be trained on data before they can be
used. In PyMVPA, all these transformations are :mod:`~mvpa.mappers`.

Let's create a dummy dataset (5 samples, 12 features). This time we will use a
new method to create the dataset, the `dataset_wizard`. Here it is fully
equivalent to a regular constructor call (i.e.  `~mvpa.datasets.base.Dataset`),
but we will shortly see some nice convenience aspects

  >>> from mvpa.suite import *
  >>> ds = dataset_wizard(N.ones((5, 12)))
  >>> ds.shape
  (5, 12)

A mapper is a :term:`dataset attribute`, hence is stored in the
corresponding attribute collection. However, not every dataset actually has
a mapper. For example, the simple one we have just created doesn't have any:

  >>> 'mapper' in ds.a
  False

Now let's look at a very similar dataset that only differs in a tiny but
very important detail:

  >>> ds = dataset_wizard(N.ones((5, 4, 3)))
  >>> ds.shape
  (5, 12)
  >>> 'mapper' in ds.a
  True
  >>> print ds.a.mapper
  <FlattenMapper>

We see that the resulting dataset is identical to the one above, but this time
it got created from a 3D samples array (i.e. five samples, where each is a 4x3
matrix). Somehow this 3D array got transformed into a 2D samples array in the
dataset. This magic behavior is unveiled by looking that the dataset's mapper
-- a `~mvpa.mappers.flatten.FlattenMapper`.

The purpose of this mapper is precisely what we have just observed: reshaping
data arrays into 2D. It does it by preserving the first axis (in PyMVPA datasets
this is the axis that separates the samples) and concatenates all other axis
into the second one.

A very important feature of this mapper is that this transformation is
reversible. We can simply ask the mapper to put our samples back into the
original 3D shape.

  >>> orig_data = ds.a.mapper.reverse(ds.samples)
  >>> orig_data.shape
  (5, 4, 3)


#We have learned that mappers transform a dataset, so let's transform it and
#look what happens. A very simple trasnformation is choosing a subset of the
#features (i.e. slicing):
#
#  >>> myids = [1, 2, 8, 10]
#  >>> subds = ds[:, myids]
#  >>> subds.shape
#  (5, 4)
#  >>> 'mapper' in subds.a
#  True
#  >>> print subds.a.mapper
#  <ChainMapper: <Flatten>-<FeatureSlice>>


Doing get_haxby2001_data() From Scratch
=======================================

.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.mappers
     ~mvpa.mappers.base.Mapper
