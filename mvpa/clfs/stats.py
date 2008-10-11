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
from mvpa.misc.state import Stateful, StateVariable

if __debug__:
    from mvpa.base import debug


class Nonparametric(object):
    """Non-parametric distribution -- derives cdf based on stored values.

    Introduced to complement parametric distributions present in scipy.stats.
    """

    def __init__(self, dist_samples):
        self._dist_samples = dist_samples


    @staticmethod
    def fit(dist_samples):
        return [dist_samples]


    def cdf(self, x):
        """Returns the frequency/probability of a value `x` given the estimated
        distribution. Returned values are determined left or right tailed
        depending on the constructor setting.

        In case a `FeaturewiseDatasetMeasure` was used to estimate the
        distribution the method returns an array. In that case `x` can be
        a scalar value or an array of a matching shape.
        """
        return (self._dist_samples <= x).mean(axis=0)


class NullHyp(Stateful):
    """Base class for null-hypothesis testing.

    """

    _ATTRIBUTE_COLLECTIONS = ['states']

    def __init__(self, tail='left', **kwargs):
        """Cheap initialization.

        :Parameter:
          tail: str ['left', 'right', 'any']
            Which tail of the distribution to report. For 'any' it chooses
            the tail it belongs to based on the comparison to p=0.5
        """
        Stateful.__init__(self, **kwargs)

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


    def p(self, x):
        """Given `tail` provide a p value using cdf()
        """
        is_scalar = N.isscalar(x)
        if is_scalar:
            x = [x]

        cdf = self.cdf(x)
        if self._tail == 'left':
            pass
        elif self._tail == 'right':
            cdf = 1 - cdf
        elif self._tail == 'any':
            right_tail = (cdf >= 0.5)
            cdf[right_tail] = 1.0 - cdf[right_tail]

        if is_scalar: return cdf[0]
        else:         return cdf


class MCNullHyp(NullHyp):
    """Null-hypothesis distribution is estimated from randomly permutted data labels.

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

    _DEV_DOC = """
    TODO automagically decide on the number of samples/permutations needed
    Caution should be paid though since resultant distributions might be
    quite far from some conventional ones (e.g. Normal) -- it is expected to
    have them bimodal (or actually multimodal) in many scenarios.
    """

    dist_samples = StateVariable(enabled=False,
                                 doc='Samples obtained for each permutation')

    def __init__(self, dist_class=Nonparametric, permutations=100, **kwargs):
        """Cheap initialization.

        :Parameter:
          dist_class: class
            This can be any class which provides parameters estimate
            using `fit()` method to initialize the instance, and
            provides `cdf(x)` method for estimating value of x in CDF.
            All distributions from SciPy's 'stats' module can be used.
          permutations: int
            This many permutations of label will be performed to
            determine the distribution under the null hypothesis.
        """
        NullHyp.__init__(self, **kwargs)

        self._dist_class = dist_class
        self._dist = []                 # actual distributions

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

        # decide on the arguments to measure
        if not vdata is None:
            measure_args = [vdata, wdata]
        else:
            measure_args = [wdata]

        # estimate null-distribution
        for p in xrange(self.__permutations):
            # new permutation all the time
            # but only permute the training data and keep the testdata constant
            #
            # TODO this really needs to be more clever! If data samples are
            # shuffled within a class it really makes no difference for the
            # classifier, hence the number of permutations to estimate the
            # null-distribution of transfer errors can be reduced dramatically
            # when the *right* permutations (the ones that matter) are done.
            wdata.permuteLabels(True, perchunk=False)

            # compute and store the measure of this permutation
            # assume it has `TransferError` interface
            dist_samples.append(measure(*measure_args))

        # restore original labels
        wdata.permuteLabels(False, perchunk=False)

        # store samples
        self.dist_samples = dist_samples = N.asarray(dist_samples)

        # fit distribution per each element

        # to decide either it was done on scalars or vectors
        shape = dist_samples.shape
        nshape = len(shape)
        # if just 1 dim, original data was scalar, just create an
        # artif dimension for it
        if nshape == 1:
            dist_samples = dist_samples[:, N.newaxis]

        # Nonparametric is actually generic enough, but scipy.stats
        # are not multivariate, thus we need to fit per each element.
        # XXX could be more elegant?
        dist_samples_rs = dist_samples.reshape((shape[0], -1))
        dist = []
        for samples in dist_samples_rs.T:
            params = self._dist_class.fit(samples)
            if __debug__ and 'STAT' in debug.active:
                debug('STAT', 'Estimated parameters for the %s are %s'
                      % (self._dist_class, str(params)))
            dist.append(self._dist_class(*params))
        self._dist = dist


    def cdf(self, x):
        """Return value of the cumulative distribution function at `x`.
        """
        if self._dist is None:
            # XXX We might not want to descriminate that way since
            # usually generators also have .cdf where they rely on the
            # default parameters. But then what about Nonparametric
            raise RuntimeError, "Distribution has to be fit first"

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
        return N.array(cdfs)


    def clean(self):
        """Clean stored distributions

        Storing all of the distributions might be too expensive
        (e.g. in case of Nonparametric), and the scope of the object
        might be too broad to wait for it to be destroyed. Clean would
        bind dist_samples to empty list to let gc revoke the memory.
        """
        self._dist = []



# XXX think how to come up with some generic decorator
#     to do deprecation warning
class MCNullDist(MCNullHyp):
    def __init__(self, *args, **kwargs):
        warning('Deprecation: class %s is superseeded by %s'
                % (self.__class__.__name__, self.__class__.__bases__[0].__name__))
        MCNullHyp.__init__(self, *args, **kwargs)


# XXX I would not even mind to absorb this functionality to be default
#     within NullHyp
class FixedNullHyp(NullHyp):
    """Proxy/Adaptor class for SciPy distributions.

    All distributions from SciPy's 'stats' module can be used with this class.

    >>> import numpy as N
    >>> from scipy import stats
    >>> from mvpa.clfs.stats import FixedNullHyp
    >>>
    >>> dist = FixedNullHyp(stats.norm(loc=2, scale=4))
    >>> dist.p(2)
    array(0.5)
    >>>
    >>> dist.cdf(N.arange(5))
    array([ 0.30853754,  0.40129367,  0.5       ,  0.59870633,  0.69146246])
    >>>
    >>> dist = FixedNullHyp(stats.norm(loc=2, scale=4), tail='right')
    >>> dist.p(N.arange(5))
    array([ 0.69146246,  0.59870633,  0.5       ,  0.40129367,  0.30853754])
    """
    def __init__(self, dist, **kwargs):
        """
        :Parameter:
          dist: distribution object
            This can be any object the has a `cdf()` method to report the
            cumulative distribition function values.
        """
        NullHyp.__init__(self, **kwargs)

        self._dist = dist


    def fit(self, measure, wdata, vdata=None):
        """Does nothing since the distribution is already fixed."""
        pass


    def cdf(self, x):
        """Return value of the cumulative distribution function at `x`.
        """
        return self._dist.cdf(x)

