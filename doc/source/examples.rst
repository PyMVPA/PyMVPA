.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: example

.. _chap_examples:

****************************
Example Analyses and Scripts
****************************

Each of the examples in this section is a stand-alone script containing all
necessary code to run some analysis. All examples are shipped with PyMVPA and
can be found in the `doc/examples/` directory in the source package. This
directory might include some more special-interest examples which are not listed
here.

Some examples need to access a sample dataset available in the `data/`
directory within the root of the PyMVPA hierarchy, and thus have to be invoked
directly from PyMVPA root (e.g. `doc/examples/searchlight.py`).
Alternatively, one can download a full example dataset, which is explained in
the next section.

Preprocessing
=============

.. toctree::

   examples/projections
   examples/smellit


Analysis
========

.. toctree::

   examples/hyperplane_demo
   examples/start_easy
   examples/smlr
   examples/clfs_examples
   examples/gpr
   examples/gpr_model_selection0
   examples/cachedkernel
   examples/searchlight_minimal
   examples/searchlight
   examples/searchlight_dsm
   examples/searchlight_surf
   examples/sensanas
   examples/svdclf
   examples/svm_margin
   examples/permutation_test
   examples/nested_cv
   examples/match_distribution
   examples/eventrelated
   examples/hyperalignment
   examples/mdp_mnist
   examples/eyemovements


Visualization
=============

.. toctree::

   examples/erp_plot
   examples/knn_plot
   examples/pylab_2d
   examples/topo_plot
   examples/som
   examples/mri_plot


Miscellaneous
=============

.. toctree::

   examples/skl_transformer_demo
   examples/skl_classifier_demo
   examples/skl_regression_demo
   examples/kerneldemo
   examples/curvefitting
