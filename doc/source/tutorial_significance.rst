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

Previously :ref:`while looking at classification <chap_tutorial_classifiers>`
we observed that classification error depends on the chosen classification
method, data preprocessing, and how the error was obtained -- training error
vs generalization estimates using different data splitting strategies.
Moreover in :ref:`attempts to localize activity using searchlight
<chap_tutorial_searchlight>` we saw that generalization error can reach
relatively small values even when processing random data which has no true
signal.  So, the value of the error alone does not provide sufficient evidence
to state that our classifier or any other method actually learnt some
mapping from the data into variables of interest.

Researchers interested in developing statistical learning methods usually aim
at achieving as high generalization performance as possible.  New published
methods often stipulate their advantage over existing ones by comparing their
generalization performance on publicly available datasets with known
characteristics (number of classes, independence of samples, actual presence
of information of interest, etc).  Therefore, generalization performances
presented in statistical learning publications are usually high enough to
obliterate even a slight chance that they obtained such performance simply by
chance.  For example, those classifiers trained on _MNIST dataset of
handwritten digits were worth reporting whenever they demonstrated average
errors of only 1-2% while doing classification among samples of 10 different
digits (the largest error reported was 12% using the simplest classification
approach).

.. _MNIST: http://yann.lecun.com/exdb/mnist

The situation is substantially different in the domain of neural data
analysis.  There classification is most often used not to construct a reliable
mapping from data into behavioral variable(s) with as small error as possible,
but rather to show that learnt classifier seems to provide mapping good enough
to claim that such mapping exists.  Such existence claim is conventionally
verified with a classical methodology of null-hypothesis significance testing (NHST),
whenever achievement of generalization performance excursion above
"chance-level" is taken for the proof that data carries effects of interest.
NHST Such approach goes in-line with traditional statistical data analysis in
behavioral and medical sciences, although it has been widely criticized for
mixing up 
What differs neural datasets:

- relatively ***low number of samples***, especially in the domain of fMRI data
  analysis
- ***large dimensionality***
- data are ***noisy***
- data often consists of ***non-independent samples***
- ***unknown ground-truth*** (either there is an effect at all, or if there is --
  what is inherent bias/error)
- ***nature of signal is poorly understood*** like in the case of fMRI BOLD.

Let's investigate effects of those factors on classification performance with
simple examples...

What differs multivariate analysis from univariate

- avoids ***multiple comparisons*** problem in NHST
- has higher ***flexibility***, thus lower ***stability***

Multivariate methods became very popular in the last decade partially due to
their inherent ability to avoid multiple comparisons issue, which is a flagman
of difficulties while going for a "fishing expedition" with univariate
methods.  Performing cross-validation on entire ROI or even full-brain allowed
people to state presence of so desired effects without defending chosen
critical value against multiple-comparisons.  Unfortunately, as there is no
such thing as "free lunch", ability to work with all observable data at once
came at a price for multivariate methods. ...


Whenever low number of samples

it seems to be important to have reasonable methodology to assess reliable ways ...

.. note:: Intro: Statistical learning brought into the realm of hypothesis testing

.. todo:: Literature search for what other domains such approach is also used

.. note:: Statistical learning is about constructing reliable models to
          describe the data, and not really to reason either data is noise.



Experimental Design
===================

.. note:: "Randomization is a crucial aspect of experimental design" (NH02),
          show reincarnated and improved (incorporate SequenceStats)
          Dataset.summary()

 can't be done when

 - dependent variable is assessed after data has been collected (RT, ACC, etc)


Hypothesis Testing
==================

.. note:: goal: p(H0|Data), H0-test gives p(Data|H0)

.. note:: ways to assess by-chance distribution -- from fixed, to
          estimated parametric, to non-parametric permutation testing
		  Try to provide an example where even non-parametric is overly
		  optimistic (if it is, as it is in Yarik's head ;-))

.. index:: monte-carlo, permutation

 would blind permutation be enough? nope... permutation testing holds whenever:
   - exchangeability

NH02:
"Applications of permutation testing methods to single subject fMRI require modelling the temporal auto-correlation in the time series."



Statistical Treatment of Sensitivities
======================================

.. note:: how do we decide to threshold sensitivities, remind them searchlight
          results with strong bimodal distributions, distribution outside of
          the brain as a true by-chance.  May be reiterate that sensitivities
          of bogus model are bogus

Moreover, constructed mapping with barely "above-chance" performance is often
further analyzed for its :ref:`sensitivity to the input variables
<chap_tutorial_sensitivity>`.




References
==========

:ref:`Cohen, J. (1994) <Coh94>`
  *Classical critic of null hypothesis significance testing*

:ref:`Nichols et al. (2002) <NH02>`
  *Overview of standard nonparametric randomization and permutation testing
  applied to neuroimaging data (e.g. fMRI)*

:ref:`Ioannidis, J. (2005) <Ioa05>`
  *Simulation study speculating that it is more likely for a research claim to
   be false than true.  Along the way the paper highlights aspects to keep in
   mind while assessing the 'scientific significance' of any given study, such
   as, viability, reproducibility, and results.*

.. only:: html

  .. autosummary::
     :toctree: generated

     ~numpy.ndarray
     ~scipy.stats.distributions.norm
     ~mvpa.clfs.stats.Nonparametric
     ~mvpa.clfs.stats.rv_semifrozen
     ~mvpa.clfs.stats.FixedNullDist
     ~mvpa.clfs.stats.MCNullDist

