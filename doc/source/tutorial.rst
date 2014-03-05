.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial:

*******************************
Tutorial Introduction to PyMVPA
*******************************

In this tutorial we are going to take a look at all major parts of PyMVPA,
introduce the most important concepts, and explore particular functionality in
real-life analysis examples. This tutorial also serves as basic course
material for workshops on introductions to MVPA. Please contact us, if you are
interested in hosting a PyMVPA workshop at your institution.

Please note that this tutorial is only concerned with aspects directly related
to PyMVPA.  It does **not** teach basic Python_ programming. If you are new to
Python, it is recommend that you take a look at the
:ref:`chap_tutorial_prerequisites` for information about what you should know
and how to obtain that knowledge.

Throughout the tutorial there will be little exercises with tasks that are
meant to deepen the understanding of a particular problem or to train
important skills. However, even without a dedicated exercise the reader is
advised to run the tutorial code interactively and explore code snippets
beyond what is touched by the tutorial. Typically, only the most important
aspects will be mentioned and each building block in PyMVPA can be used in
more flexible ways than what is shown. Enjoy the ride.

Through the course of the tutorial we will analyze :ref:`real BOLD fMRI data
<datadb_tutorial_data>`. Therefore, to be able to run the code in this
tutorial, you need to download the :ref:`corresponding data from the PyMVPA
website <datadb_tutorial_data>`. Once downloaded, extract the tarball.  On a
NeuroDebian-enabled system, the tutorial data is also available from the
``python-mvpa2-tutorialdata`` package.

The ``pymvpa2-tutorial`` command (installed with PyMVPA) can be invoked in a
console in order to launch a tutorial session. If the tutorial data was
downloaded manually it may be necessary to specify the appropriate
``--tutorial-data-path`` option (see ``pymvpa2-tutorial --help`` for more
information).

Virtually every Python script starts with some ``import`` statements that load
functionality provided elsewhere. Likewise a tutorial session need to import
the PyMVPA packages and some little helpers we are going to use in the
tutorial::

>>> from mvpa2.tutorial_suite import *

If this command succeeds without error, everything is ready to go.

If you want to prevent yourself from re-typing all code snippets into the
terminal window, you might want to investigate IPython's ``%cpaste``
command, or use the provided `IPython notebooks`_ for each tutorial part.

.. _Python: http://www.python.org
.. _IPython notebook: http://ipython.org/notebook

.. toctree::
   :maxdepth: 2

   tutorial_prerequisites
   tutorial_datasets
   tutorial_mappers
   tutorial_classifiers
   tutorial_searchlight
   tutorial_meta_classifiers
   tutorial_sensitivity
   tutorial_eventrelated
   tutorial_eventrelated_searchlight
   tutorial_significance
