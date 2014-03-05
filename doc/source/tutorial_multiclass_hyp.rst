.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_multiclass_hyp:

**************************
Part X: Hypothesis testing
**************************

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
procedure the chosen classifiers makes correct predictions for approximately
half of the data points. The big question is now: **What does that tell us?**

There are many scenarios that could lead to this prediction performance. It
could be that the fitted classifier model is very good, but only capture the
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
can imagine to run such an analysis on data from different parts of the brain
and the results change -- without necessarily having a big impact on the
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
likelihood of any of our hypothesis being true or false. It is trivial to do a
Chi-square test of the confusion table...

>>> print 'Chi^2: %.3f (p=%.3f)' % cv.ca.stats.stats["CHI^2"]
Chi^2: 1942.519 (p=0.000)

... but, again, it doesn't tell us anything other than the classifier not just
doing random guesses. It would be much more useful, if we could estimate how
likely it is, given the observed confusion matrix, that the employed classifier
is able to discriminate *all* stimulus classes from each other, and not just a
subset. Even more useful would be, if we could relate this probability to
specific alternative hypotheses, such as an animate/inanimate-only distinction.

:ref:`Olivetti et al. (2012) <OGA2012>` have devised a method that allows for
doing exactly that. The confusion matrix is analyzed in a Bayesian framework
regarding the statistical dependency of observed and predicted class labels.
Confusions within a set of classes that cannot be discriminated should be
independently distributed, while there should be a statistical dependency of
confusion patterns within any set of classes that can all be discriminated from
each other.

This algorithm is available in the
:class:`mvpa2.clfs.transerror.BayesConfusionHypothesis` node.

>>> cv = CrossValidation(clf, NFoldPartitioner(),
...                      errorfx=None,
...                      postproc=ChainNode((Confusion(labels=ds.UT),
...                                          BayesConfusionHypothesis())))
>>> # cv_results = cv(ds)
>>> # print cv_results.fa.stat  ['log(p(C|H))' 'log(p(H|C))']

Most likely hypothesis to explain this confusion matrix

print cv_results.sa.hypothesis[np.argsort(cv_results.samples[:,1])[-1]]
[['bottle'], ['cat'], ['chair'], ['face'], ['house'], ['scissors'], ['scrambledpix'], ['shoe']]

References
==========
