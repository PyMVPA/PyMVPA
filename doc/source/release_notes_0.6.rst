.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. index:: release notes
.. _chap_release_notes_0.6:

***************************
Release Notes -- PyMVPA 0.6
***************************

* All mappers, classifiers, regressions, and measure are now implemented as
  :class:`Node`\s that can be called with a class:`Dataset` and return a
  processed dataset.  All nodes provide a ``generate()`` method that causes the
  node to yields to processing result. In addition, there is special nodes that
  implement more complex generators yielding multiple results (e.g. resampling,
  permuting, or splitting nodes).

* Splitters as such do not exist any longer. They have been replaced by a number
  of generators that offer the same functionality, but can be combined in more
  flexible way.

* There is no `TransferError` anymore. Any classifier (or measure) can evaluate
  error functions using a post-processing node, e.g.::

     SVM(..., postproc=BinaryFxNode(mean_mismatch_error, 'targets')

  This is possible, because by default classifiers simply return a dataset with
  all predictions as samples and assign the actual targets as a samples
  attribute. The post-processing node can subsequently compute arbitrary error
  values (total, or per-unique sample id, ...).

* Feature selections are no longer a separate entity, but are implemented as
  training procedures for `SliceMapper`. The most important effect is
  that SliceMapper tries to perform slicing without copying whenever
  possible. Note that any feature selection procedure has to be trained on a
  dataset before it can be used -- otherwise it won't do the right thing(TM).
