#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Estimator for classifier error distributions."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base import externals, warning

if __debug__:
    from mvpa.base import debug

class Distribution(object):
    def __init__(self, tail='left'):
        """Cheap initialization.

        :Parameter:
          tail: str ['left', 'right', 'any']
            Which tail of the distribution to report. For 'any' it chooses
            the tail it belongs to based on the comparison to p=0.5
        """
        self._tail = tail

        # sanity check
        if self._tail not in ['left', 'right', 'any']:
            raise ValueError, 'Unknown value "%s" to `tail` argument.' \


    def fit(self, measure, wdata, vdata=None):
        """Implement to fit the distribution to the data."""
        raise NotImplementedError


    def cdf(self, x):
        """Implementations return the value of the cumulative distribution
        function (left or right tail dpending on the setting).
        """
        raise NotImplementedError



class MCDist(Distribution):
    """Base class of a bunch of distributions for Null testing, where
    parameters are estimated from the data

    The distribution is estimated by calling fit() with an appropriate
    `DatasetMeasure` or `TransferError` instance and a training and a
    validation dataset (in case of a `TransferError`). For a customizable
    amount of cycles the training data labels are permuted and the
    corresponding measure computed. In case of a `TransferError` this is the
    error when predicting the *correct* labels of the validation dataset.

    The distribution can be queried using the `cdf()` method, which can be
    configured to report probabilities/frequencies from `left` or `right` tail,
    i.e. fraction of the distribution that is lower or larger than some
    critical value.

    This class also supports `FeaturewiseDatasetMeasure`. In that case `cdf()`
    returns an array of featurewise probabilities/frequencies.
    """

    def __init__(self, permutations=100, **kwargs):
        """Cheap initialization.

        :Parameter:
            permutations: int
                This many classification attempts with permuted label vectors
                will be performed to determine the distribution under the null
                hypothesis.
        """
        Distribution.__init__(self, **kwargs)

        self._dist_samples = None
        self.__permutations = permutations
        """Number of permutations to compute the estimate the null
        distribution."""


    def fit(self, measure, wdata, vdata=None):
        """Fit the distribution by performing multiple cycles which repeatedly
        permuted labels in the training dataset.

        :Parameter:
            measure: (`Featurewise`)`DatasetMeasure` | `TransferError`
                TransferError instance used to compute all errors.
            wdata: `Dataset` which gets permuted and used to compute the
                measure/transfer error multiple times.
            vdata: `Dataset` used for validation.
                If provided measure is assumed to be a `TransferError` and
                working and validation dataset are passed onto it.
        """
        dist_samples = []
        """Holds the values for randomized labels."""

        # estimate null-distribution
        for p in xrange(self.__permutations):
            # new permutation all the time
            # but only permute the training data and keep the testdata constant
            # TODO this really needs to be more clever! If data samples are
            # shuffled within a class it really makes no difference for the
            # classifier, hence the number of permutations to estimate the
            # null-distribution of transfer errors can be reduced dramatically
            # when the *right* permutations (the ones that matter) are done.
            wdata.permuteLabels(True, perchunk=False)

            # compute and store the measure of this permutation
            if not vdata is None:
                # assume it has `TransferError` interface
                dist_samples.append(measure(vdata, wdata))
            else:
                dist_samples.append(measure(wdata))

        # store errors
        self._dist_samples = N.asarray(dist_samples)

        # restore original labels
        wdata.permuteLabels(False, perchunk=False)


    def clean(self):
        """Clean stored data

        Storing all of the distribution samples might be too
        expensive, and the scope of the object might be too broad to
        wait for it to be destroyed. Clean would bind dist_samples to
        empty list to let gc revoke the memory.
        """
        self._dist_samples = []


    @property
    def dist_samples(self):
        """Samples obtained by permutting the labels"""
        return self._dist_samples

#
# XXX whole logic left/right/any can be done in 1 place
# (Distribution?) if all subclasses simply estimate values of cdf for
# given x.  That would result in a better logic: MCNonparamDist would
# be just MCFixedDist(dist_gen=Nonparam), so everything needs further
# refactoring imho (yoh).
#
# Also cdf(x) per se shouldn't do 'left/right' -- that is a job of .p(x)
# I guess ;-)
#
class MCNonparamDist(MCDist):
    """Class to determine the distribution of a measure under the NULL
    distribution (no signal).

    No assumptions are made about the shape of the distribution under the null
    hypothesis. Instead this distribution is estimated by performing multiple
    measurements with permuted `label` vectors, hence no or random signal.
    """
    def cdf(self, x):
        """Returns the frequency/probability of a value `x` given the estimated
        distribution. Returned values are determined left or right tailed
        depending on the constructor setting.

        In case a `FeaturewiseDatasetMeasure` was used to estimate the
        distribution the method returns an array. In that case `x` can be
        a scalar value or an array of a matching shape.
        """
        if self._tail == 'left':
            return (self._dist_samples <= x).mean(axis=0)
        elif self._tail == 'right':
            return (self._dist_samples >= x).mean(axis=0)
        elif self._tail == 'any':
            # easy if just scalar
            if N.isscalar(x):
                right_tail = N.median(self._dist_samples) < x
                if right_tail:
                    return (self._dist_samples >= x).mean(axis=0)
                else:
                    return (self._dist_samples <= x).mean(axis=0)

            # now handle case of 'x is sequence'
            x = N.array(x)

            # determine on which tail we are
            # if critical is larger than median of distribution:
            right_tail = N.array(N.median(self._dist_samples) < x) #, axis=0))
            # ancient numpy does not have axis kwarg for median

            # generate container for results
            res = N.zeros(right_tail.shape)

            # catch special cases of all right and all left
            right_tail_fraction = right_tail.mean()

            # handle right tail cases
            if right_tail_fraction > 0:
                res[right_tail] = (
                    self._dist_samples[:, right_tail] >= x[right_tail]
                        ).mean(axis=0)

            # handle left tail cases
            if right_tail_fraction < 1:
                left_tail = right_tail == False
                res[left_tail] = (
                    self._dist_samples[:, left_tail] <= x[left_tail]
                        ).mean(axis=0)

            return res

        else:
            raise RuntimeError, "This should not happen!"


# XXX think how to come up with some generic decorator
#     to do this:
class MCNullDist(MCNonparamDist):
    def __init__(self, *args, **kwargs):
        warning('Deprecation: class %s is superseeded by %s'
                % (self.__class__.__name__, self.__class__.__bases__[0].__name__))
        MCNonparamDist.__init__(self, *args, **kwargs)


# some bogus class which never matches
class _rv_frozen_bogus(object):
    pass

if externals.exists('scipy'):
    import scipy.stats
    rv_frozen = scipy.stats.distributions.rv_frozen
else:
    rv_frozen = _rv_frozen_bogus


class MCFixedDist(MCDist):
    """Proxy/Adaptor class for SciPy distributions which first estimates the distribution.

    All distributions from SciPy's 'stats' module can be used with this class.

    TODO automagically decide on the number of samples/permutations needed
    Caution should be paid though since resultant distributions might be
    quite far from some conventional ones (e.g. Normal) -- it is expected to
    have them bimodal (or actually multimodal) in many scenarios.

    >>> import numpy as N
    >>> from scipy import stats
    >>> from mvpa.clfs.stats import MCFixedDist
    >>>
    >>> dist = MCFixedDist(stats.norm)
    >>> dist.cdf(2)
    array(0.5)
    >>>
    >>> dist.cdf(N.arange(5))
    array([ 0.30853754,  0.40129367,  0.5       ,  0.59870633,  0.69146246])
    >>>
    >>> dist = MCFixedDist(stats.norm, tail='right')
    >>> dist.cdf(N.arange(5))
    array([ 0.69146246,  0.59870633,  0.5       ,  0.40129367,  0.30853754])
    """
    def __init__(self, dist_gen, **kwargs):
        """
        :Parameter:
          dist_gen: distribution generator object
            This can be any generator the has a `fit()` method to report the
            cumulative distribition function values.
        """
        MCDist.__init__(self, **kwargs)

        self._dist_gen = dist_gen
        self._dist = None


    def fit(self, *args, **kwargs):
        """Does nothing since the distribution is already fixed."""
        super(MCFixedDist, self).fit(*args, **kwargs)
        dist_samples = self.dist_samples

        # to decide either it was done on scalars or vectors
        shape = dist_samples.shape
        nshape = len(shape)
        # if just 1 dim, original data was scalar, just create an
        # artif dimension for it
        if nshape == 1:
            dist_samples = dist_samples[:, N.newaxis]

        # we need to fit per each element
        # XXX could be more elegant?
        dist_samples_rs = dist_samples.reshape((shape[0], -1))
        dist = []
        for samples in dist_samples_rs.T:
            params = self._dist_gen.fit(samples)
            if __debug__:
                debug('STAT', 'Estimated parameters for the %s are %s'
                      % (self._dist_gen, str(params)))
            dist.append(self._dist_gen(*params))
        self._dist = dist


    def cdf(self, x):
        """Return value of the cumulative distribution function at `x`.
        """
        if self._dist is None:
            # XXX We might not want to descriminate that way since
            # usually generators also have .cdf where they rely on the
            # default parameters
            raise RuntimeError, "Distribution has to be estimated first"
        #raise NotImplementedError

        is_scalar = N.isscalar(x)
        if is_scalar:
            x = [x]

        # assure x is a 1D array now
        x = N.asanyarray(x).reshape((-1,))

        if len(self._dist) != len(x):
            raise ValueError, 'Distribution was fit for structure with %d' \
                  ' elements, whenever now queried with %d elements' \
                  % (len(self._dist), len(x))

        # extract cdf values per each element
        cdfs = [ dist.cdf(v) for v, dist in zip(x, self._dist) ]
        cdfs = N.array(cdfs)

        # XXX probably is very similar to what FixedDist.cdf would
        # look like if it was complete... may be we could unite those
        # two somehow
        if self._tail == 'left':
            res = cdfs
        elif self._tail == 'right':
            res = 1 - cdfs
        elif self._tail == 'any':
            # more fun
            right_tail = (cdfs >= 0.5)
            res = cdfs
            res[right_tail] = 1.0 - res[right_tail]

        if is_scalar:
            return res[0]
        return res


class FixedDist(Distribution):
    """Proxy/Adaptor class for SciPy distributions.

    All distributions from SciPy's 'stats' module can be used with this class.

    >>> import numpy as N
    >>> from scipy import stats
    >>> from mvpa.clfs.stats import FixedDist
    >>>
    >>> dist = FixedDist(stats.norm(loc=2, scale=4))
    >>> dist.cdf(2)
    array(0.5)
    >>>
    >>> dist.cdf(N.arange(5))
    array([ 0.30853754,  0.40129367,  0.5       ,  0.59870633,  0.69146246])
    >>>
    >>> dist = FixedDist(stats.norm(loc=2, scale=4), tail='right')
    >>> dist.cdf(N.arange(5))
    array([ 0.69146246,  0.59870633,  0.5       ,  0.40129367,  0.30853754])
    """
    def __init__(self, dist, **kwargs):
        """
        :Parameter:
          dist: distribution object
            This can be any object the has a `cdf()` method to report the
            cumulative distribition function values.
        """
        Distribution.__init__(self, **kwargs)

        self._dist = dist


    def fit(self, measure, wdata, vdata=None):
        """Does nothing since the distribution is already fixed."""
        pass


    def cdf(self, x):
        """Return value of the cumulative distribution function at `x`.
        """
        cdf = self._dist.cdf(x)
        if self._tail == 'left':
            return cdf
        elif self._tail == 'right':
            return 1 - cdf
        elif self._tail == 'any':
            right_tail = (cdf >= 0.5)
            cdf[right_tail] = 1.0 - cdf[right_tail]
            return cdf

