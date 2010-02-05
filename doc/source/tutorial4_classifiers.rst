.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial4:

**********************************************
Part 4: Classifier -- All Alike, Yet Different
**********************************************

What is

* prediction
* estimate
* states, parameters
* meta classifiers...

.. exercise::

   Try doing the Z-Scoring beforce computing the mean samples per category.
   What happens to the generalization performance of the classifier?
   ANSWER: It becomes 100%!


.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.clfs.base.Classifier
