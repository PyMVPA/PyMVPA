#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Monte-Carlo testing of Classifier-based Analyses
================================================

.. index:: statistical testing, monte-carlo, permutation

It is often desirable to be able to make statements like *"Performance is
significantly above chance-level"* and to help with that PyMVPA supports *Null*
hypothesis (aka *H0*) testing for any :class:`~mvpa2.measures.base.Measure`.
Measures take an optional constructor argument ``null_dist`` that can be used
to provide an instance of some :class:`~mvpa2.clfs.stats.NullDist` estimator.
If the properties of the expected *Null* distribution are known a-priori, it is
possible to use any distribution specified in SciPy's ``stats`` module for this
purpose (see e.g. :class:`~mvpa2.clfs.stats.FixedNullDist`).

However, as with other applications of statistics in classifier-based analyses
there is the problem that we typically do not know the distribution of a
variable like error or performance under the *Null* hypothesis (i.e. the
probability of a result given that there is no signal), hence we cannot easily
assign the adored p-values. Even worse, the chance-level or guess probability
of a classifier depends on the content of a validation dataset, e.g. balanced
or unbalanced number of samples per label and total number of labels.

One approach to deal with this situation is to *estimate* the *Null*
distribution using permutation testing. The *Null* distribution is then
estimated by computing the measure of interest multiple times using original
data samples but with permuted targets.  Since quite often the exploration of
all permutations is unfeasible, Monte-Carlo testing (see :ref:`Nichols
et al. (2006) <NH02>`) allows to obtain stable estimate with only a
limited number of random permutations.

Given the results computed using permuted targets one can now determine the
probability of the empirical result (i.e. the one computed from the
original training dataset) under the *no signal* condition. This is
simply the fraction of results from the permutation runs that is
larger or smaller than the empirical (depending on whether one is
looking at performances or errors).

Here is how this looks for a simple cross-validated classification in PyMVPA.
We start by generated a dataset with 200 samples and 3 features of which 2 carry
some relevant signal.
"""

# lazy import
from mvpa2.suite import *

# enable progress output for MC estimation
if __debug__:
    debug.active += ["STATMC"]

# some example data with signal
ds = normal_feature_dataset(perlabel=100, nlabels=2, nfeatures=3,
                            nonbogus_features=[0,1], snr=0.3, nchunks=2)

"""
Now we can start collecting the pieces that play a role in this analysis. We
need a classifier.
"""

clf = LinearCSVMC()

"""
We need a :term:`generator` than will produce partitioned datasets, one for each
fold of the cross-validation. A partitioned dataset is basically the same as the
original dataset, but has an additional samples attribute that indicates whether
particular samples will be the *part* of the data that is used for training the
classifier, or for testing it. By default, the
:class:`~mvpa2.generators.partition.NFoldPartitioner` will create a sample
attribute ``partitions`` that will label one :term:`chunk` in each fold
differently from all others (hence mark it as taken-out for testing).
"""

partitioner = NFoldPartitioner()

"""
We need two pieces for the Monte Carlo shuffling. The first of them is
an instance of an
:class:`~mvpa2.generators.permutation.AttributePermutator` that will
permute the target attribute of the dataset for each iteration.  We
will instruct it to perform 200 permutations. In a real analysis the
number of permutations should be larger to get stable estimates.
"""

permutator = AttributePermutator('targets', count=200)

"""
The second mandatory piece for a Monte-Carlo-style estimation of
the *Null* distribution is the actual "estimator".
:class:`~mvpa2.clfs.stats.MCNullDist` will use the
constructed ``permutator`` to shuffle the targets and later on report
p-value from the left tail of the *Null* distribution, because we are
going to compute errors and are interested in them being *lower* than
chance. Finally we also ask for all results from Monte-Carlo shuffling
to be stored for subsequent visualization of the distribution.
"""

distr_est = MCNullDist(permutator, tail='left', enable_ca=['dist_samples'])

"""

Now we have all pieces and can conduct the actual cross-validation. We assign
a post-processing :term:`node` ``mean_sample`` that will take care of averaging
error values across all cross-validation fold. Consequently, the *Null*
distribution of *average cross-validated classification error* will be estimated
and used for statistical evaluation.
"""

cv = CrossValidation(clf, partitioner,
                     errorfx=mean_mismatch_error,
                     postproc=mean_sample(),
                     null_dist=distr_est,
                     enable_ca=['stats'])
# run
err = cv(ds)

"""
Now we have a usual cross-validation error and ``cv`` stores
:term:`conditional attribute`s such as confusion matrices`:
"""

print 'CV-error:', 1 - cv.ca.stats.stats['ACC']

"""
However, in addition it also provides the results of the statistical
evaluation. The :term:`conditional attribute` ``null_prob`` has a
dataset that contains the p-values representing the likelihood of an
error equal or lower to the output one under the *Null* hypothesis,
i.e. no actual relevant signal in the data. For a reason that will
appear sensible later on, the p-value is contained in a dataset.
"""

p = cv.ca.null_prob
# should be exactly one p-value
assert(p.shape == (1,1))
print 'Corresponding p-value:',  np.asscalar(p)

"""
We can now look at the distribution of the errors under *H0* and plot the
expected chance level as well as the empirical error.
"""

# make new figure
pl.figure()
# histogram of all computed errors from permuted data
pl.hist(np.ravel(cv.null_dist.ca.dist_samples), bins=20)
# empirical error
pl.axvline(np.asscalar(err), color='red')
# chance-level for a binary classification with balanced samples
pl.axvline(0.5, color='black', ls='--')
# scale x-axis to full range of possible error values
pl.xlim(0,1)
pl.xlabel('Average cross-validated classification error')

"""
We can see that the *Null* or chance distribution is centered around the
expected chance-level and the empirical error value is in the far left tail,
thus unlikely to belong to *Null* distribution, and hence the low p-value.

This could be the end, but sometimes one needs to have a closer look. Let's say your
data is not that homogeneous. Let's say that some :term:`chunk <Chunk>` may be very
different from others. You might want to look at the error value probability for
specific cross-validation folds. Sounds complicated? Luckily it is very simple.
It only needs a tiny change in the cross-validation setup -- the removal of the
``mean_sample`` post-processing :term:`node`.
"""

cv = CrossValidation(clf, partitioner,
                     errorfx=mean_mismatch_error,
                     null_dist=distr_est,
                     enable_ca=['stats'])
# run
err = cv(ds)

assert (err.shape == (2,1))
print 'CV-errors:', np.ravel(err)

"""
Now we get two errors -- one for each cross-validation fold and
most importantly, we also get the two associated p-values.
"""

p = cv.ca.null_prob
assert(p.shape == (2,1))
print 'Corresponding p-values:',  np.ravel(p)

"""
What happened is that a dedicated *Null* distribution has been estimated for
each element in the measure results. Without ``mean_sample`` an error is
reported for each CV-fold, hence a separate distributions are estimated for
each CV-fold too. And because we have also asked for all distribution samples
to be reported, we can now plot both distribution and both empirical errors.
But how do we figure out with value is which?

As mentioned earlier all results are returned in Datasets. All datasets have
compatible sample and feature axes, hence corresponding elements.
"""

assert(err.shape == p.shape == cv.null_dist.ca.dist_samples.shape[:2])

# let's make a function this time
def plot_cv_results(cv, err, title):
    # make new figure
    pl.figure()
    colors = ['green', 'blue']
    # null distribution samples
    dist_samples = np.asarray(cv.null_dist.ca.dist_samples)
    for i, e in enumerate(err):
        # histogram of all computed errors from permuted data per CV-fold
        pl.hist(np.ravel(dist_samples[i]), bins=20, color=colors[i],
                label='CV-fold %i' %i, alpha=0.5,
                range=(dist_samples.min(), dist_samples.max()))
        # empirical error
        pl.axvline(np.asscalar(e), color=colors[i])

    # chance-level for a binary classification with balanced samples
    pl.axvline(0.5, color='black', ls='--')
    # scale x-axis to full range of possible error values
    pl.xlim(0,1)
    pl.xlabel(title)

plot_cv_results(cv, err, 'Per CV-fold classification error')

"""
We have already seen that the statistical evaluation is pretty flexible.
However, we haven't yet seen whether it is flexible enough. To illustrate that
think about what was done in the above Monte Carlo analyses.

A dataset was shuffled repeatedly, and for each iteration a full
cross-validation of classification error was performed. However, the shuffling
was done on the *full* dataset, hence target were permuted in both training
*and* testing dataset portions in each CV-fold. This basically means that for
each Monte Carlo iteration the classifier was tested on a new data/signal.
However, we may be more interested in what the classifier has to say on the
*actual* data, but when it was trained on randomly permuted data.

As you can guess this is possible too and goes like this. The most important
difference is that we are going to use  now a dedicate measure to estimate the
*Null* distribution. That measure is very similar to the cross-validation we
have used before, but differs in an important bit. Let's look at the pieces.
"""

# how often do we want to shuffle the data
repeater = Repeater(count=200)
# permute the training part of a dataset exactly ONCE
permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1)
# CV with null-distribution estimation that permutes the training data for
# each fold independently
null_cv = CrossValidation(
            clf,
            ChainNode([partitioner, permutator], space=partitioner.get_space()),
            errorfx=mean_mismatch_error)
# Monte Carlo distribution estimator
distr_est = MCNullDist(repeater, tail='left', measure=null_cv,
                       enable_ca=['dist_samples'])
# actual CV with H0 distribution estimation
cv = CrossValidation(clf, partitioner, errorfx=mean_mismatch_error,
                     null_dist=distr_est, enable_ca=['stats'])

"""
The ``repeater`` is a simple node that returns any given dataset a
configurable number of times. We use the helper to configure the number of
Monte Carlo iterations. The new ``permutator`` is again configured to shuffle
the ``targets`` attribute, but only *once* and only for samples that were
labeled as being part of the training set in a particular CV-fold (the
``partitions`` sample attribute will be created by the NFoldPartitioner that we
have configured earlier).

The most important difference is a new dedicated measure that will be used to
perform a cross-validation analysis under the *H0* hypotheses. To this end
we set up a standard CV procedure with a twist: we use a chained generator
(comprising of the typical partitioner and the new one-time permutator).
This will cause the CV to permute the training set for each CV-fold internally
(and that is what we wanted).

Now we assign the *H0* cross-validation procedure to the distribution
estimator and use the ``repeater`` to set the number of iterations. Lastly, we
plug everything into a standard CV analysis with, again, a non-permuting
``partitioner`` and the pimped *Null* distribution estimator.

Now we just need to run it, and plot the results the same way we did before.
"""

err = cv(ds)
print 'CV-errors:', np.ravel(err)
p = cv.ca.null_prob
print 'Corresponding p-values:',  np.ravel(p)
# plot
plot_cv_results(cv, err,
                'Per CV-fold classification error (only training permutation)')

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()

"""

There a many ways to futher tweak the statistical evaluation. For example, if
the family of the distribution is known (e.g. Gaussian/Normal) and provided
with the ``dist_class`` parameter of ``MCNullDist``, then permutation tests
done by ``MCNullDist`` allow determining the distribution parameters. Under the
(strong) assumption of Gaussian distribution, 20-30 permutations should be
sufficient to get sensible estimates of the distribution parameters.

But that would be another story...

"""
