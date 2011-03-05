.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
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
of information of interest, etc.).  Therefore, generalization performances
presented in statistical learning publications are usually high enough to
obliterate even a slight chance that they could have been obtained  simply by
chance.  For example, those classifiers trained on MNIST_ dataset of
handwritten digits were worth reporting whenever they demonstrated average
**errors of only 1-2%** while doing classification among samples of 10 different
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
whenever the achievement of generalization performance with *statistically
significant* excursion away from the *chance-level* is taken as the proof that
data carries effects of interest.

The main conceptual problem with NHST is a widespread belief that having observed
the data, the level of significance at which H0 could be rejected is equivalent to the
probability of the H0 being true.  I.e. if it is unlikely that data comes from
H0, it is as unlikely for H0 being true.  Such assumptions were shown to be
generally wrong using :ref:`deductive and Bayesian reasoning <Coh94>` since
P(D|H0) not equal P(H0|D) (unless P(D)==P(H0)).  Moreover, *statistical
significance* alone, taken without accompanying support on viability and
reproducibility of a given finding, was argued :ref:`more likely to be
incorrect <Ioa05>`.

..
   exerciseTODO::

   If results were obtained at the same significance p<0.05, which finding
   would you believe to reflect the existing phenomenon: ability to decode
   finger-tapping sequence of the subject participating in the experiment or
   ability to decode ...

What differs multivariate analysis from univariate is that it

* avoids **multiple comparisons** problem in NHST
* has higher **flexibility**, thus lower **stability**

Multivariate methods became very popular in the last decade of neuroimaging
research partially due to their inherent ability to avoid multiple comparisons
issue, which is a flagman of difficulties while going for a *fishing
expedition* with univariate methods.  Performing cross-validation on entire
ROI or even full-brain allowed people to state presence of so desired effects
without defending chosen critical value against multiple-comparisons.
Unfortunately, as there is no such thing as *free lunch*, ability to work with
all observable data at once came at a price for multivariate methods.

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

In the following part of the tutorial we will investigate the effects of some
of those factors on classification performance with simple (or not so)
examples.  But first lets overview the tools and methodologies for NHST
commonly employed.


Statistical Tools in Python
===========================

`scipy` Python module is an umbrella project to cover the majority of core
functionality for scientific computing in Python.  In turn, :mod:`~scipy.stats`
submodule covers a wide range of continuous and discrete distributions and
statistical functions.

.. exercise::

  Glance over the `scipy.stats` documentation for what statistical functions
  and distributions families it provides.  If you feel challenged, try to
  figure out what is the meaning/application of :func:`~scipy.stats.rdist`.

The most popular distribution employed for NHST in the context of statistical
learning, is :class:`~scipy.stats.binom` for testing either generalization
performance of the classifier on independent data could provide evidence that
the data contains the effects of interest.

.. note::

   `scipy.stats` provides function :func:`~scipy.stats.binom_test`, but that
   one was devised only for doing two-sides tests, thus is not directly
   applicable for testing generalization performance where we aim at the tail
   with lower than chance performance values.

.. exercise::

   Think about scenarios when could you achieve strong and very significant
   mis-classification performance, i.e. when, for instance, binary classifier
   tends to generalize into the other category.  What could it mean?

:class:`~scipy.stats.binom` whenever instantiated with the parameters of the
distribution (which are number of trials, probability of success on each
trial), it provides you ability to easily compute a variety of statistics of
that distribution.  For instance, if we want to know, what would be the probability of having achieved
57 of more correct responses out of 100 trials, we need to use a survival
function (1-cdf) to obtain the *weight* of the right tail including 57
(i.e. query for survival function of 56):

>>> from scipy.stats import binom
>>> binom100 = binom(100, 1./2)
>>> print '%.3g' % binom100.sf(56)
0.0967

Apparently obtaining 57 correct out 100 cannot be considered significantly
good performance by anyone.  Lets investigate how many correct responses we
need to reach the level of 'significance' and use *inverse survival function*:

>>> binom100.isf(0.05) + 1
59.0
>>> binom100.isf(0.01) + 1
63.0

So, depending on your believe and prior support for your hypothesis and data
you should get at least 59-63 correct responses from a 100 trials to claim
the existence of the effects.  Someone could rephrase above observation that to
achieve significant performance you needed an effect size of 9-13
correspondingly for those two levels of significance.

.. exercise::

  Plot a curve of *effect sizes* (number of correct predictions above
  chance-level) vs a number of trials at significance level of 0.05 for a range
  of trial numbers from 4 to 1000.  Plot %-accuracy vs number of trials for
  the same range in a separate plot. TODO

.. XXX ripples...
.. nsamples = np.arange(4, 1000, 2)
.. effect_sizes = [ceil(binom(n,0.5).isf(0.05) + 1 - n/2) for n in nsamples]
.. pl.plot(nsamples, effect_sizes)
.. pl.figure()
.. pl.plot(nsamples, 0.5 + effect_sizes / nsamples)
.. pl.ylabel('Accuracy to reach p<=0.05')
.. pl.hlines([0.5, 1.0], 0, 1000)

..
  commentTODO::

  If this is your first ever analysis and you are not comparing obtained
  results across different models (classifiers), since then you would
  (theoretically) correct your significance level for multiple comparisons.


Dataset Exploration for Confounds
=================================

:ref:`"Randomization is a crucial aspect of experimental design... In the
absence of random allocation, unforeseen factors may bias the results."
<NH02>`.

Unfortunately it is impossible to detect and warn about all possible sources
of confounds which would invalidate NHST based on a simple parametric binomial
test.  As a first step, it is always useful to inspect your data for possible
sources of samples non-independence, especially if your results are not
strikingly convincing or too provocative.  Possible obvious problems could be:

 * dis-balanced testing sets (usually non-equal number of samples for each
   label in any given chunk of data)
 * order effects: either preference of having samples of particular target
   in a specific location or the actual order of targets

To allow for easy inspection of dataset to prevent such obvious confounds,
:func:`~mvpa.datasets.miscfx.summary` function (also a method of any
`Dataset`) was constructed.  Lets have yet another look at our 8-categories
dataset:

>>> from tutorial_lib import *
>>> # alt: `ds = load_tutorial_results('ds_haxby2001')`
>>> ds = get_haxby2001_data(roi='vt')
>>> print ds.summary()
Dataset: 16x577@float64, <sa: chunks,targets,time_coords,time_indices>, <fa: voxel_indices>, <a: imghdr,imgtype,mapper,voxel_dim,voxel_eldim>
stats: mean=11.5788 std=13.7772 var=189.811 min=-49.5554 max=97.292
<BLANKLINE>
Counts of targets in each chunk:
      chunks\targets     bottle cat chair face house scissors scrambledpix shoe
                           ---  ---  ---   ---  ---     ---        ---      ---
0.0+2.0+4.0+6.0+8.0+10.0    1    1    1     1    1       1          1        1
1.0+3.0+5.0+7.0+9.0+11.0    1    1    1     1    1       1          1        1
<BLANKLINE>
Summary for targets across chunks
    targets  mean std min max #chunks
   bottle      1   0   1   1     2
     cat       1   0   1   1     2
    chair      1   0   1   1     2
    face       1   0   1   1     2
    house      1   0   1   1     2
  scissors     1   0   1   1     2
scrambledpix   1   0   1   1     2
    shoe       1   0   1   1     2
<BLANKLINE>
Summary for chunks across targets
          chunks         mean std min max #targets
0.0+2.0+4.0+6.0+8.0+10.0   1   0   1   1      8
1.0+3.0+5.0+7.0+9.0+11.0   1   0   1   1      8
Sequence statistics for 16 entries from set ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
Counter-balance table for orders up to 2:
Targets/Order O1                |  O2                |
   bottle:     0 2 0 0 0 0 0 0  |   0 0 2 0 0 0 0 0  |
     cat:      0 0 2 0 0 0 0 0  |   0 0 0 2 0 0 0 0  |
    chair:     0 0 0 2 0 0 0 0  |   0 0 0 0 2 0 0 0  |
    face:      0 0 0 0 2 0 0 0  |   0 0 0 0 0 2 0 0  |
    house:     0 0 0 0 0 2 0 0  |   0 0 0 0 0 0 2 0  |
  scissors:    0 0 0 0 0 0 2 0  |   0 0 0 0 0 0 0 2  |
scrambledpix:  0 0 0 0 0 0 0 2  |   1 0 0 0 0 0 0 0  |
    shoe:      1 0 0 0 0 0 0 0  |   0 1 0 0 0 0 0 0  |
Correlations: min=-0.52 max=1 mean=-0.067 sum(abs)=5.7

You can see that labels were balanced across chunks -- i.e. that each chunk
has an equal number of samples of each target label, and that samples of
different labels are evenly distributed across chunks.  TODO...

Counter-balance table shows either there were any order effects among
conditions.  In this case we had only two instances of each label in the
dataset due to the averaging of samples across blocks, so it would be more
informative to look at the original sequence.  To do so avoiding loading a
complete dataset we would simply provide the stimuli sequence to
:class:`~mvpa.clfs.miscfx.SequenceStats` for the analysis:

>>> attributes_filename = os.path.join(tutorial_data_path, 'data', 'attributes.txt')
>>> attr = SampleAttributes(attributes_filename)
>>> targets = np.array(attr.targets)
>>> ss = SequenceStats(attr.targets)
>>> print ss
Sequence statistics for 1452 entries from set ['bottle', 'cat', 'chair', 'face', 'house', 'rest', 'scissors', 'scrambledpix', 'shoe']
Counter-balance table for orders up to 2:
Targets/Order O1                           |  O2                           |
   bottle:    96  0  0  0  0  12  0  0  0  |  84  0  0  0  0  24  0  0  0  |
     cat:      0 96  0  0  0  12  0  0  0  |   0 84  0  0  0  24  0  0  0  |
    chair:     0  0 96  0  0  12  0  0  0  |   0  0 84  0  0  24  0  0  0  |
    face:      0  0  0 96  0  12  0  0  0  |   0  0  0 84  0  24  0  0  0  |
    house:     0  0  0  0 96  12  0  0  0  |   0  0  0  0 84  24  0  0  0  |
    rest:     12 12 12 12 12 491 12 12 12  |  24 24 24 24 24 394 24 24 24  |
  scissors:    0  0  0  0  0  12 96  0  0  |   0  0  0  0  0  24 84  0  0  |
scrambledpix:  0  0  0  0  0  12  0 96  0  |   0  0  0  0  0  24  0 84  0  |
    shoe:      0  0  0  0  0  12  0  0 96  |   0  0  0  0  0  24  0  0 84  |
Correlations: min=-0.19 max=0.88 mean=-0.00069 sum(abs)=77

Order statistics look funky at first, but they would not surprise you if you
recall the original design of the experiment -- blocks of 8 TRs per each
category, interleaved with 6 TRs of rest condition.  Since samples from two
adjacent blocks are far apart enough not to contribute to 2-back table (O2
table on the right), it is worth inspecting if there was any dis-balance in
the order of the picture conditions blocks.  It would be easy to check if we
simply drop the 'rest' condition from consideration:

>>> print SequenceStats(targets[targets != 'rest'])
Sequence statistics for 864 entries from set ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'scrambledpix', 'shoe']
Counter-balance table for orders up to 2:
Targets/Order O1                       |  O2                       |
   bottle:    96  2  1  2  2  3  0  2  |  84  4  2  4  4  6  0  4  |
     cat:      2 96  1  1  1  1  4  2  |   4 84  2  2  2  2  8  4  |
    chair:     2  3 96  1  1  2  1  2  |   4  6 84  2  2  4  2  4  |
    face:      0  3  3 96  1  1  2  2  |   0  6  6 84  2  2  4  4  |
    house:     0  1  2  2 96  2  4  1  |   0  2  4  4 84  4  8  2  |
  scissors:    3  0  2  3  1 96  0  2  |   6  0  4  6  2 84  0  4  |
scrambledpix:  2  1  1  2  3  2 96  1  |   4  2  2  4  6  4 84  2  |
    shoe:      3  2  2  1  3  0  1 96  |   6  4  4  2  6  0  2 84  |
Correlations: min=-0.3 max=0.87 mean=-0.0012 sum(abs)=59

TODO

.. exercise::

   Generate few 'designs' consisting of varying condition sequences and assess
   their counter-balance.  Generate some random designs using random number
   generators or permutation functions provided in :mod:`numpy.random` and
   assess their counter-balance.

..
   exerciseTODO::

   If you take provided data set, what accuracy could(would) you achieve in
   Taro-reading of the future stimuli conditions based on just previous
   stimuli condition(fMRI data) data 15-30 seconds prior the actual stimuli
   block?  Would it be statistically/scientifically significant?

Some sources of confounds might be hard to detect or to eliminate:

 - dependent variable is assessed after data has been collected (RT, ACC,
   etc) so it might be hard to guarantee equal sampling across different
   splits of the data.

 - motion effects, if motion is correlated with the design, might introduce
   major confounds into the signal.  With multivariate analysis the problem
   becomes even more sever due to the high sensitivity of multivariate methods
   and the fact that motion effects might be impossible to eliminate entirely
   since they are strongly non-linear.  So, even if you regress out whatever
   number of descriptors describing motion (mean displacement, angles, shifts,
   etc.) you would not be able to eliminate motion effects entirely.  And that
   residual variance from motion spread through the entire volume might
   contribute to your *generalization performance*.

.. exercise::

   Inspect the arguments of generic interface of all splitters
   :class:`~mvpa.datasets.splitters.Splitter` for a possible workaround in the
   case of dis-balanced targets.

Therefore, before the analysis on the actual fMRI data, it might be worth
inspecting what kind of :term:`generalization` performance you might obtain if
you operate simply on the confounds (e.g. motion parameters and effects).

.. index:: monte-carlo, permutation


Hypothesis Testing
==================

.. note::

  When thinking about what critical value to choose for NHST keep such
  :ref:`guidelines from NHST inventor, Dr.Fisher <Fis25>` in mind.  For
  significance range '0.2 - 0.5' he says: "judged significant, though barely
  so; ... these data do not, however, demonstrate the point beyond possibility
  of doubt".

Ways to assess *by-chance* null-hypothesis distribution of measures range from
fixed, to estimated parametric, to non-parametric permutation testing.
Unfortunately not a single way provides an ultimate testing facility to be
applied blindly to any chosen problem without investigating the
appropriateness of the data at hand (see previous section).  Every kind of
:class:`~mvpa.measures.base.Measure` provides an easy way to trigger
assessment of *statistical significance* by specifying ``null_dist`` parameter
with a distribution estimator.  After a given measure is computed, the
corresponding p-value(s) for the returned value(s) could be accessed at
``ca.null_prob``.

:ref:`"Applications of permutation testing methods to single subject fMRI
require modelling the temporal auto-correlation in the time series." <NH02>`

.. exercise::

   Try to assess significance of the finding on two problematic categories
   from 8-categories dataset without averaging the samples within the blocks
   of the same target.  Even non-parametric test should be overly optimistic
   (forgotten **exchangeability** requirement for parametric testing, such as
   multiple samples within a block for a block design)... TODO


Independent Samples
-------------------

Since "voodoo correlations" paper, most of the literature in brain imaging is
seems to became more careful in avoiding "double-dipping" and keeping their
testing data independent from training data, which is one of the major
concerns for doing valid hypothesis testing later on.  Not much attention is
given though to independence of samples aspect -- i.e. not only samples in
testing set should be independent from training ones, but, to make binomial
distribution testing valid, testing samples should be independent from each
other as well.  The reason is simple -- number of the testing samples defines
the width of the null-chance distribution, but consider the limiting case
where all testing samples are heavily non-independent, consider them to be a
1000 instances of the same sample.  Canonical binomial distribution would be
very narrow, although effectively it is just 1 independent sample being
tested, thus ... TODO



Statistical Treatment of Sensitivities
======================================

.. note:: Statistical learning is about constructing reliable models to
          describe the data, and not really to reason either data is noise.

.. note:: How do we decide to threshold sensitivities, remind them searchlight
          results with strong bimodal distributions, distribution outside of
          the brain as a true by-chance.  May be reiterate that sensitivities
          of bogus model are bogus

Moreover, constructed mapping with barely *above-chance* performance is often
further analyzed for its :ref:`sensitivity to the input variables
<chap_tutorial_sensitivity>`.



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

