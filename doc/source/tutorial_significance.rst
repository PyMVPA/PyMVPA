.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_significance:

**************************************************
Part 8: The Earth Is Round -- Significance Testing
**************************************************

.. note:: Intro: Statistical learning brought into the realm of hypothesis testing
.. todo:: Literature search for what other domains such approach is also used

.. note:: Statistical learning is about constructing reliable models to
          describe the data, and not really to reason either data is noise.


Hypothesis Testing
==================

.. note:: goal: p(H0|Data), H0-test gives p(Data|H0)

.. note:: ways to assess by-chance distribution -- from fixed, to
          estimated parametric, to non-parametric permutation testing
		  Try to provide an example where even non-parametric is overly
		  optimistic (if it is, as it is in Yarik's head ;-))

.. index:: monte-carlo, permutation


Statistical Treatment of Sensitivities
======================================

.. note:: how do we decide to threshold sensitivities, remind them searchlight
          results with strong bimodal distributions, distribution outside of
          the brain as a true by-chance.  May be reiterate that sensitivities
          of bogus model are bogus



References
==========

:ref:`Cohen, J. (1994) <Coh94>`
  *Classical critic of null hypothesis significance testing*

:ref:`Nichols et al. (2002) <NH02>`
  *Overview of standard nonparametric randomization and permutation testing
  applied to neuroimaging data (e.g. fMRI)*


.. only:: html

  .. autosummary::
     :toctree: generated

     ~numpy.ndarray
     ~scipy.stats.distributions.norm
     ~mvpa.clfs.stats.Nonparametric
     ~mvpa.clfs.stats.rv_semifrozen
     ~mvpa.clfs.stats.FixedNullDist
     ~mvpa.clfs.stats.MCNullDist

