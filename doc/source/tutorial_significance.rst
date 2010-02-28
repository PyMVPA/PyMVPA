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

Previously, :ref:`while looking at classification <chap_tutorial_classifiers>`
we have observed that classification error depends on the chosen
classification method, data preprocessing, and how the error was obtained --
training error vs generalization estimates using different data splitting
strategies.  Moreover in :ref:`attempts to localize activity using searchlight
<chap_tutorial_searchlight>` we saw that generalization error can reach
relatively small values even when processing random data which (should) have
no true signal.  So, the value of the error alone does not provide
sufficient evidence to state that our classifier or any other method actually
learnt the mapping from the data into variables of interest.  So, how do we
decide what estimate of error can provide us sufficient evidence that
constructed mapping reflects the underlying phenomenon or that our data
carried the signal of interest?

Researchers interested in developing statistical learning methods usually aim
at achieving as high generalization performance as possible.  Newly published
methods often stipulate their advantage over existing ones by comparing their
generalization performance on publicly available datasets with known
characteristics (number of classes, independence of samples, actual presence
of information of interest, etc).  Therefore, generalization performances
presented in statistical learning publications are usually high enough to
obliterate even a slight chance that they could have been obtained  simply by
chance.  For example, those classifiers trained on _MNIST dataset of
handwritten digits were worth reporting whenever they demonstrated average
errors of only 1-2% while doing classification among samples of 10 different
digits (the largest error reported was 12% using the simplest classification
approach).

.. _MNIST: http://yann.lecun.com/exdb/mnist

.. Statistical learning brought into the realm of hypothesis testing

.. todo:: Literature search for what other domains such approach is also used

The situation is substantially different in the domain of neural data
analysis.  There classification is most often used not to construct a reliable
mapping from data into behavioral variable(s) with as small error as possible,
but rather to show that learnt mapping is good enough to claim that such
mapping exists and data carries the effects caused by the corresponding
experiment.  Such an existence claim is conventionally verified with a
classical methodology of null-hypothesis (H0) significance testing (NHST),
whenever achievement of generalization performance excursion away from
"chance-level" is taken as the proof that data carries effects of interest.

The conceptual problem with NHST is a widespread belief that having observed
the data, the level of significance H0 could be rejected is equivalent to the
probability of the H0 being true.  I.e. if it is unlikely that data comes from
H0, it is as unlikely for H0 being true.  Such assumptions were shown to be
generally wrong using :ref:`deductive and Bayesian reasoning <Coh94>` since
P(D|H0) not equal P(H0|D) (unless P(D)==P(H0)).  Moreover, "statistical
significance" alone, taken without accompanying support on viability and
reproducibility of a given finding, was argued :ref:`more likely to be
incorrect <Ioa05>`.


.. exerciseTODO::

   If results were obtained at the same significance p<0.05, which finding
   would you believe to reflect the existing phenomenon: ability to decode
   finger-tapping sequence of the subject participating in the experiment or
   ability to decode ...

The second peculiarity of the application of statistical learning in
psychological research is the actual neural data which researchers are doomed
to analyze.  As we have already seen from previous tutorial parts, typical
fMRI data has

- relatively **low number of samples** (up to few thousands in total)
- relatively **large dimensionality** (tens of thousands)
- **small signal-to-noise ratio**
- **non-independent measurements**
- **unknown ground-truth** (either there is an effect at all, or if there is --
  what is inherent bias/error)
- **unknown nature of the signal**, since BOLD effect is not entirely
     understood.

Lets investigate effects of some of those factors on classification
performance with simple examples.  But first lets overview the tools and
methodologies for NHST commonly employed.


Statistical Tools in Python
===========================

`scipy` Python module is an umbrella project to cover the majority of core
functionality for scientific computing.  In turn, :mod:`~scipy.stats`
submodule covers a wide range of continuous and discrete distributions and
statistical functions.

.. exercise::

  Glance over the `scipy.stats` documentation for what statistical functions
  and distributions families it provides.  If you feel challenged, try to
  figure out what is the meaning/application of :func:`~scipy.stats.rdist`.

The most popular distribution employed in carrying out NHST in the context
of statistical learning, is :func:`~scipy.stats.binom` for testing either
generalization performance of the classifier on independent data could provide
evidence that the data contains the effects of interest.  Lets see how


.. index:: monte-carlo, permutation

Hypothesis Testing
==================

.. note::

  When thinking about what critical value to choose for NHST keep such
  :ref:`guidelines from NHST inventor, Dr.Fisher <Fis25>` in mind.  For
  significance range '0.2 - 0.5' he says: "judged significant, though barely
  so; ... these data do not, however, demonstrate the point beyond possibility
  of doubt".


.. note::

   Ways to assess by-chance distribution -- from fixed, to estimated
   parametric, to non-parametric permutation testing Try to provide an
   example where even non-parametric is overly optimistic (if it is, as it is
   in Yarik's head ;-))


Would blind permutation be enough? nope... permutation testing holds whenever
**exchangeability** could be guaranteed.

NH02: "Applications of permutation testing methods to single subject fMRI
require modelling the temporal auto-correlation in the time series."


Effects of Experimental Design
==============================

:ref:`"Randomization is a crucial aspect of experimental design" <NH02>`.

.. todo:: show reincarnated and improved (incorporate SequenceStats)
          Dataset.summary()

 can't be done when

 - dependent variable is assessed after data has been collected (RT, ACC, etc)



Statistical Treatment of Sensitivities
======================================

.. note:: Statistical learning is about constructing reliable models to
          describe the data, and not really to reason either data is noise.

.. note:: how do we decide to threshold sensitivities, remind them searchlight
          results with strong bimodal distributions, distribution outside of
          the brain as a true by-chance.  May be reiterate that sensitivities
          of bogus model are bogus

Moreover, constructed mapping with barely "above-chance" performance is often
further analyzed for its :ref:`sensitivity to the input variables
<chap_tutorial_sensitivity>`.

What differs multivariate analysis from univariate

- avoids **multiple comparisons** problem in NHST
- has higher **flexibility**, thus lower **stability**

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





References
==========

:ref:`Cohen, J. (1994) <Coh94>`
  *Classical critic of null hypothesis significance testing*

:ref:`Fisher, R. A. (1925) <Fis25>`
  *One of the 20th century's most influential books on statistical methods, which
  coined the term 'Test of significance'.*

:ref:`Ioannidis, J. (2005) <Ioa05>`
  *Simulation study speculating that it is more likely for a research claim to
  be false than true.  Along the way the paper highlights aspects to keep in
  mind while assessing the 'scientific significance' of any given study, such
  as, viability, reproducibility, and results.*

:ref:`Nichols et al. (2002) <NH02>`
  *Overview of standard nonparametric randomization and permutation testing
  applied to neuroimaging data (e.g. fMRI)*

:ref:`Wright, D. (2009) <Wri09>`
  *Historical excurse into the life of 10 prominent statisticians of XXth century
  and their scientific contributions.*

.. only:: html

  .. autosummary::
     :toctree: generated

     ~numpy.ndarray
     ~scipy.stats.distributions.norm
     ~mvpa.clfs.stats.Nonparametric
     ~mvpa.clfs.stats.rv_semifrozen
     ~mvpa.clfs.stats.FixedNullDist
     ~mvpa.clfs.stats.MCNullDist

