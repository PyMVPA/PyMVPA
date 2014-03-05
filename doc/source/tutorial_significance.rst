.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, statistical testing, monte-carlo, permutation
.. _chap_tutorial_significance:

***********************************************
WiP: The Earth Is Round -- Significance Testing
***********************************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_significance.ipynb>`_]

*Null* hypothesis testing
=========================

It is often desirable to be able to make statements like *"Performance is
significantly above chance-level"*. To help with that PyMVPA supports *Null*
hypothesis (aka *H0*) testing for any :class:`~mvpa2.measures.base.Measure`.

However, as with other applications of statistics in classifier-based analyses,
there is the problem that we typically do not know the distribution of a
variable like error or performance under the *Null* hypothesis (i.e. the
probability of a result given that there is no signal), hence we cannot easily
assign the adored p-values. Even worse, the chance-level or guess probability
of a classifier depends on the content of a validation dataset, e.g. balanced
or unbalanced number of samples per label, total number of labels, as well as
the peculiarities of "independence" of training and testing data -- especially
in the neuroimaging domain.

Monte Carlo -- here I come!
---------------------------

One approach to deal with this situation is to *estimate* the *Null*
distribution using permutation testing. The *Null* distribution is
estimated by computing the measure of interest multiple times using the original
data samples but with permuted targets, presumably scrambling or destroying the
signal of interest.  Since quite often the exploration of all permutations is
unfeasible, Monte-Carlo testing (see :ref:`Nichols et al. (2002) <NH02>`)
allows us to obtain a stable estimate with only a limited number of random
permutations.

Given the results computed using permuted targets, we can now determine the
probability of the empirical result (i.e. the one computed from the original
training dataset) under the *no signal* condition. This is simply the fraction
of results from the permutation runs that is larger or smaller than the
empirical (depending on whether one is looking at performances or errors).

Here we take a look at how this is done for a simple cross-validated
classification in PyMVPA.  We start by generating a dataset with 200 samples
and 3 features of which only two carry some relevant signal. Afterwards we set
up a standard leave-one-chunk-out cross-validation procedure for an SVM
classifier. At this point we have seen this numerous times, and the code should
be easy to read:

>>> # lazy import
>>> from mvpa2.suite import *
>>> # some example data with signal
>>> ds = normal_feature_dataset(perlabel=100, nlabels=2, nfeatures=3,
...                             nonbogus_features=[0,1], snr=0.3, nchunks=2)
>>> # classifier
>>> clf = LinearCSVMC()
>>> # data folding
>>> partitioner = NFoldPartitioner()
>>> # complete cross-validation setup
>>> cv = CrossValidation(clf,
...                      partitioner,
...                      postproc=mean_sample(),
...                      enable_ca=['stats'])
>>> err = cv(ds)

.. exercise::

  Take a look at the performance statistics of the classifier. Explore how it
  changes with different values of the signal-to-noise (``snr``) parameter
  of the dataset generator function.

Now we want to run this analysis again, repeatedly and with a fresh
permutation of the targets for each run. We need two pieces for the Monte
Carlo shuffling. The first is an instance of an
:class:`~mvpa2.generators.permutation.AttributePermutator` that will
permute the target attribute of the dataset for each iteration.  We
will instruct it to perform 200 permutations. In a real analysis, the
number of permutations will often be more than that.

>>> permutator = AttributePermutator('targets', count=200)

.. exercise::

  The ``permutator`` is a generator. Try generating all 200 permuted
  datasets.

The second necessary component for a Monte-Carlo-style estimation of the *Null*
distribution is the actual "estimator".  :class:`~mvpa2.clfs.stats.MCNullDist`
will use the already created ``permutator`` to shuffle the targets and later on
report the p-value from the left tail of the *Null* distribution, because we are
going to compute errors and we are interested in them being *lower* than chance.
Finally, we also ask for all results from Monte-Carlo shuffling to be stored for
subsequent visualization of the distribution.

>>> distr_est = MCNullDist(permutator, tail='left', enable_ca=['dist_samples'])

The rest is easy. Measures take an optional constructor argument ``null_dist``
that can be used to provide an instance of some
:class:`~mvpa2.clfs.stats.NullDist` estimator -- and we have just created one!
Because a cross-validation is nothing but a measure, we can assign it our *Null*
distribution estimator, and it will also perform permutation testing, in
addition to the regular classification analysis on the "real" dataset.
Consequently, the code hasn't changed much:

>>> cv_mc = CrossValidation(clf,
...                         partitioner,
...                         postproc=mean_sample(),
...                         null_dist=distr_est,
...                         enable_ca=['stats'])
>>> err = cv_mc(ds)
>>> cv.ca.stats.stats['ACC'] == cv_mc.ca.stats.stats['ACC']
True

Other than it taking a bit longer to compute, the performance did not change.
But the additional waiting wasn't in vain, as we get the results of the
statistical evaluation. The ``cv_mc`` :term:`conditional attribute`
``null_prob`` has a dataset that contains the p-values representing the
likelihood of an empirical value (i.e. the result from analysing the original
dataset) being equal or lower to one under the *Null* hypothesis, i.e. no
actual relevant signal in the data. Or in more concrete terms, the p-value
is the fraction of permutation results less than or
equal to the empirical result.


>>> p = cv_mc.ca.null_prob
>>> # should be exactly one p-value
>>> p.shape
(1, 1)
>>> np.asscalar(p) < 0.1
True

.. exercise::

  How many cross-validation analyses were computed when running ``cv_mc``?
  Make sure you are not surprised that it is more than 200.
  What is the minimum p-value that we can get from 200 permutations?

Let's practise our visualization skills a bit and create a quick plot to
show the *Null* distribution and how "significant" our
empirical result is. And let's make a function for plotting to show off
our Python-foo!

>>> def make_null_dist_plot(dist_samples, empirical):
...     pl.hist(dist_samples, bins=20, normed=True, alpha=0.8)
...     pl.axvline(empirical, color='red')
...     # chance-level for a binary classification with balanced samples
...     pl.axvline(0.5, color='black', ls='--')
...     # scale x-axis to full range of possible error values
...     pl.xlim(0,1)
...     pl.xlabel('Average cross-validated classification error')
>>>
>>> # make new figure ('_ =' is only used to swallow unnecessary output)
>>> _ = pl.figure()
>>> make_null_dist_plot(np.ravel(cv_mc.null_dist.ca.dist_samples),
...                     np.asscalar(err))
>>> # run pl.show() if the figure doesn't appear automatically

You can see that we have created a histogram of the "distribution samples" stored
in the *Null* distribution (because we asked for it previously).  We can also
see that the *Null* or chance distribution is centered around the expected
chance-level and the empirical error value is in the far left tail, thus
relatively unlikely to be a *Null* result, hence the low-ish p-value.

This wasn't too bad, right? We could stop here. But there is this smell....

.. exercise::

  The p-value that we have just computed and the *Null* distribution we looked
  at are, unfortunately, **invalid** -- at least if we want to know how likely
  it is to obtain our **empirical** result under a no-signal condition. Can you
  figure out why?

  PS: The answer is obviously in the next section, so do not spoil your learning
  experience by reading it before you have thought about this issue!


Avoiding the trap OR Advanced magic 101
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is what went wrong: The dataset's class labels (aka targets) were shuffled
repeatedly, and for each iteration a full cross-validation of classification
error was computed. However, the shuffling was done on the *full* dataset,
hence target values were permuted in both training *and* testing dataset
portions in each CV-fold. This basically means that for each Monte Carlo
iteration the classifier was **tested** on new data/signal.
However, we are actually interested in what the classifier has to say about the
*actual* data, but when it was **trained** on randomly permuted data.

Doing a whole-dataset permutation is a common mistake with very beneficial
side-effects -- as you will see in a bit. Sadly, doing the permuting correctly (i.e.
in the training portion of the dataset only) is a bit more complicated due to
the data-folding scheme that we have to deal with. Here is how it goes:

>>> repeater = Repeater(count=200)

A ``repeater`` is a simple node that returns any given dataset a configurable
number of times. We use this helper to configure the number of Monte Carlo
iterations.

.. exercise::

  A :class:`~mvpa2.generators.base.Repeater` is also a generator. Try calling it
  with our dataset. What does it do? How can you get it to produce the 200
  datasets?

The new ``permutator`` is again configured to shuffle the ``targets``
attribute. But this time only *once* and only for samples that were labeled as
being part of the training set in a particular CV-fold. The ``partitions``
sample attribute is created by the NFoldPartitioner that we have already
configured earlier (or any other partitioner in PyMVPA for that matter).

>>> permutator = AttributePermutator('targets',
...                                  limit={'partitions': 1},
...                                  count=1)

The most significant difference is that we are now going to use a dedicate
measure to estimate the *Null* distribution. That measure is very similar
to the cross-validation we have used before, but differs in an important twist:
we use a chained generator to perform the data-folding. This chain comprises
of our typical partitioner (marks one chunk as testing data and the rest as
training, for all chunks) and the new one-time permutator. This chain-generator
causes the cross-validation procedure to permute the training data only for each
data-fold and leave the testing data untouched. Note, that we make the chain use
the ``space`` of the partitioner, to let the ``CrossValidation`` know which
samples attribute defines training and testing partitions.

>>> null_cv = CrossValidation(
...            clf,
...            ChainNode(
...                 [partitioner, permutator],
...                 space=partitioner.get_space()),
...            postproc=mean_sample())

.. exercise::

  Create a separate chain-generator and explore what it does. Remember: it is
  just a generator.

Now we create our new and improved distribution estimator. This looks similar
to what we did before, but we now use our dedicated *Null* cross-validation
measure, and run it as often as ``repeater`` is configured to estimate the
*Null* performance.

>>> distr_est = MCNullDist(repeater, tail='left',
...                        measure=null_cv,
...                        enable_ca=['dist_samples'])

On the "outside" the cross-validation measure for computing the empricial
performance estimate is 100% identical to what we have used before. All the
magic happens inside the distribution estimator.

>>> cv_mc_corr = CrossValidation(clf,
...                              partitioner,
...                              postproc=mean_sample(),
...                              null_dist=distr_est,
...                              enable_ca=['stats'])
>>> err = cv_mc_corr(ds)
>>> cv_mc_corr.ca.stats.stats['ACC'] == cv_mc.ca.stats.stats['ACC']
True
>>> cv_mc.ca.null_prob.samples <  cv_mc_corr.ca.null_prob.samples
array([[ True]], dtype=bool)

After running it we see that there is no change in the empirical performance
(great!), but our significance did suffer (poor thing!). We can take a look
at the whole picture by plotting our previous *Null* distribution estimate
and the new, improved one as an overlay.

>>> make_null_dist_plot(cv_mc.null_dist.ca.dist_samples, np.asscalar(err))
>>> make_null_dist_plot(cv_mc_corr.null_dist.ca.dist_samples, np.asscalar(err))
>>> # run pl.show() if the figure doesn't appear automatically

It should be obvious that there is a substantial difference in the two
estimates, but only the latter/wider distribution is valid!

.. exercise::

  Keep it in mind. Keep it in mind. Keep it in mind.



The following content is incomplete and experimental
====================================================


If you have a clue
------------------

There a many ways to further tweak the statistical evaluation. For example, if
the family of the distribution is known (e.g. Gaussian/Normal) and provided via
the ``dist_class`` parameter of ``MCNullDist``, then permutation tests samples
will be used to fit this particular distribution and estimate distribution
parameters. This could yield enormous speed-ups. Under the (strong) assumption
of Gaussian distribution, 20-30 permutations should be sufficient to get
sensible estimates of the distribution parameters. Fitting a normal distribution
would look like this. Actually, only a single modification is necessary
(the ``dist_class`` argument), but we will also reduce the number permutations.

>>> distr_est = MCNullDist(Repeater(count=200),
...                        dist_class=scipy.stats.norm,
...                        tail='left',
...                        measure=null_cv,
...                        enable_ca=['dist_samples'])
>>> cv_mc_norm = CrossValidation(clf,
...                              partitioner,
...                              postproc=mean_sample(),
...                              null_dist=distr_est,
...                              enable_ca=['stats'])
>>> err = cv_mc_norm(ds)
>>> distr = cv_mc_norm.null_dist.dists()[0]
>>> make_null_dist_plot(cv_mc_norm.null_dist.ca.dist_samples,
...                     np.asscalar(err))
>>> x = np.linspace(0,1,100)
>>> _ = pl.plot(x, distr.pdf(x), color='black', lw=2)


Family-friendly
---------------

When going through this chapter you might have thought: "Jeez, why do they need
to return a single p-value in a freaking dataset?" But there is a good reason
for this. Lets set up another cross-validation procedure. This one is basically
identical to the last one, except for not averaging classifier performances
across data-folds (i.e. ``postproc=mean_sample()``).

>>> cvf = CrossValidation(
...         clf,
...         partitioner,
...         null_dist=MCNullDist(
...                     repeater,
...                     tail='left',
...                     measure=CrossValidation(
...                                 clf,
...                                 ChainNode([partitioner, permutator],
...                                           space=partitioner.get_space()))
...                     )
...         )

If we run this on our dataset, we no longer get a single performance value,
but one per data-fold (chunk) instead:

>>> err = cvf(ds)
>>> len(err) == len(np.unique(ds.sa.chunks))
True

But here comes the interesting bit:

>>> len(cvf.ca.null_prob) == len(err)
True

So we get one p-value for each element in the datasets returned by the
cross-validation run. More generally speaking, the distribution estimation
happens independently for each value returned by a measure -- may this be
multiple samples, or multiple features, or both. Consequently, it is possible
to test a large variety of measure with this facility.


Evaluating multi-class classifications
======================================

So far we have mostly looked at the situation where a classifier is trying to
discriminate data from two possible classes. In many cases we can assume that a
classifier that *cannot* discriminate these two classes would perform at a
chance-level of 0.5 (ACC). If it does that we would conclude that there is no
signal of interest in the data, or our classifier of choice cannot pick it up.
However, there is a whole universe of classification problems where it is not
that simple.

Let's revisit the classification problem from :ref:`the chapter on classifiers
<chap_tutorial_classifiers>`.

>>> from mvpa2.tutorial_suite import *
>>> ds = get_haxby2001_data_alternative(roi='vt', grp_avg=False)
>>> print ds.sa['targets'].unique
['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe']
>>> clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
>>> cv = CrossValidation(clf, NFoldPartitioner(), errorfx=mean_mismatch_error,
...                      enable_ca=['stats'])
>>> cv_results = cv(ds)
>>> print '%.2f' % np.mean(cv_results)
0.53

So here we have an 8-way classification problem, and during the cross-validation
procedure the chosen classifier makes correct predictions for approximately
half of the data points. The big question is now: **What does that tell us?**

There are many scenarios that could lead to this prediction performance. It
could be that the fitted classifier model is very good, but only captures the
data variance for half of the data categories/classes. It could also be that
the classifier model quality is relatively poor and makes an equal amount of
errors for all classes. In both cases the average accuracy will be around 50%,
and most likely **highly significant**, given a chance performance of 1/8.  We
could now spend some time testing this significance with expensive permutation
tests, or making assumptions on the underlying distribution. However, that
would only give us a number telling us that the average accuracy is really
different from chance, but it doesn't help with the problem that the accuracy
really doesn't tell us much about what we are interested in.

Interesting hypotheses in the context of this dataset could be whether the data
carry a signal that can be used to distinguish brain response patterns from
animate vs.  inanimate stimulus categories, or whether data from object-like
stimuli are all alike and can only be distinguished from random noise, etc. One
can imagine running such an analysis on data from different parts of the brain
and the results changing -- without necessarily having a big impact on the
overall classification accuracy.

A lot more interesting information is available from the confusion matrix, a
contingency table showing prediction targets vs. actual predictions.

>>> print cv.ca.stats.matrix
[[36  7 18  4  1 18 15 18]
 [ 3 56  6 18  0  3  7  5]
 [ 2  2 21  0  4  0  3  1]
 [ 3 16  0 76  4  5  3  1]
 [ 1  1  6  1 97  1  4  0]
 [20  5 15  4  0 29 15 11]
 [ 0  1  0  0  0  2 19  0]
 [43 20 42  5  2 50 42 72]]

We can see a strong diagonal, but also block-like structure, and have to
realize that simply staring at the matrix doesn't help us to easily assess the
likelihood of any of our hypotheses being true or false. It is trivial to do a
Chi-square test of the confusion table...

>>> print 'Chi^2: %.3f (p=%.3f)' % cv.ca.stats.stats["CHI^2"]
Chi^2: 1942.519 (p=0.000)

... but, again, it doesn't tell us anything other than that the classifier is
not just doing random guesses. It would be much more useful if we could
estimate how likely it is, given the observed confusion matrix, that the
employed classifier is able to discriminate *all* stimulus classes from each
other, and not just a subset. Even more useful would be if we could relate
this probability to specific alternative hypotheses, such as an
animate/inanimate-only distinction.

:ref:`Olivetti et al. (2012) <OGA2012>` have devised a method that allows for
doing exactly that. The confusion matrix is analyzed in a Bayesian framework
regarding the statistical dependency of observed and predicted class labels.
Confusions within a set of classes that cannot be discriminated should be
independently distributed, while there should be a statistical dependency of
confusion patterns within any set of classes that can all be discriminated from
each other.

This algorithm is available in the
:class:`~mvpa2.clfs.transerror.BayesConfusionHypothesis` node.

>>> cv = CrossValidation(clf, NFoldPartitioner(),
...                      errorfx=None,
...                      postproc=ChainNode((Confusion(labels=ds.UT),
...                                          BayesConfusionHypothesis())))
>>> cv_results = cv(ds)
>>> print cv_results.fa.stat
['log(p(C|H))' 'log(p(H|C))']

Most likely hypothesis to explain this confusion matrix:

>>> print cv_results.sa.hypothesis[np.argsort(cv_results.samples[:,1])[-1]]
[['bottle'], ['cat'], ['chair'], ['face'], ['house'], ['scissors'], ['scrambledpix'], ['shoe']]





Previously in part 8
====================

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
:func:`~mvpa2.datasets.miscfx.summary` function (also a method of any
`Dataset`) was constructed.  Lets have yet another look at our 8-categories
dataset:

>>> from mvpa2.tutorial_suite import *
>>> # alt: `ds = load_tutorial_results('ds_haxby2001')`
>>> ds = get_haxby2001_data(roi='vt')
>>> print ds.summary()
Dataset: 16x577@float64, <sa: chunks,runtype,targets,time_coords,time_indices>, <fa: voxel_indices>, <a: imghdr,imgtype,mapper,voxel_dim,voxel_eldim>
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
:class:`~mvpa2.clfs.miscfx.SequenceStats` for the analysis:

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
   :class:`~mvpa2.datasets.splitters.Splitter` for a possible workaround in the
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
:class:`~mvpa2.measures.base.Measure` provides an easy way to trigger
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

