.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: example

.. _chap_examples:

*************
Full Examples
*************

Each of the examples in this section is a stand-alone script containing all
necessary code to run some analysis. All examples are shipped with PyMVPA and
can be found in the `doc/examples/` directory in the source package. This
directory might include some more special-interest examples which are not listed
here.

Some examples need to access a sample dataset available in the `data/`
directory within the root of the PyMVPA hierarchy, and thus have to be invoked
directly from PyMVPA root (e.g. `doc/examples/searchlight_2d.py`).
Alternatively, one can download a full example dataset, which is explained in
the next section.

.. include:: misc/exampledata.readme

Preprocessing
=============

.. toctree::

   examples/projections
   examples/smellit


Analysis
========

.. toctree::

   examples/start_easy
   examples/smlr
   examples/clfs_examples
   examples/gpr
   examples/searchlight_minimal
   examples/searchlight_2d
   examples/searchlight_dsm
   examples/sensanas
   examples/svdclf
   examples/permutation_test
   examples/match_distribution
   examples/eventrelated


Visualization
=============

.. toctree::

   examples/erp_plot
   examples/pylab_2d
   examples/topo_plot
   examples/som


Miscellaneous
=============

.. toctree::

   examples/kerneldemo
   examples/curvefitting
