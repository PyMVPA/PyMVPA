# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Estimator for classifier error distributions."""

from __future__ import with_statement   # Let's start using with

__docformat__ = 'restructuredtext'

import warnings

import numpy as np

from mvpa2.base import externals, warning
from mvpa2.base.state import ClassWithCollections, ConditionalAttribute
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.base.types import is_datasetlike
from mvpa2.datasets import Dataset

if __debug__:
    from mvpa2.base import debug

if externals.exists('scipy'):
    import scipy.stats.distributions as ssd

    def _auto_rcdf(dist):
        dist_check = dist

        # which to check for continuous/discrete
        if isinstance(dist, ssd.rv_frozen):
            dist_check = dist.dist
        if isinstance(dist_check, ssd.rv_discrete):
            # we need to count the exact matches
            rcdf = lambda x, *args: 1 - dist.cdf(x, *args) + dist.pmf(x, *args)
        elif isinstance(dist_check, ssd.rv_continuous):
            # for continuous it is just as good
            rcdf = lambda x, *args: 1 - dist.cdf(x, *args)
        elif isinstance(dist_check, Nonparametric):
            rcdf = dist.rcdf
        else:
            raise ValueError("Do not know how to get 'right cdf' for %s" % (dist,))
        return rcdf

else:
    def _auto_rcdf(dist):
        if isinstance(dist, Nonparametric):
            rcdf = dist.rcdf
        else:
            raise ValueError("Do not know how to get 'right cdf' for %s" % (dist,))
        return rcdf

class Nonparametric(object):
    """Non-parametric 1d distribution -- derives cdf based on stored values.

    Introduced to complement parametric distributions present in scipy.stats.
    """

    def __init__(self, dist_samples, correction='clip'):
        """
        Parameters
        ----------
        dist_samples : ndarray
          Samples to be used to assess the distribution.
        correction : {'clip'} or None, optional
          Determines the behavior when .cdf is queried.  If None -- no
          correction is made.  If 'clip' -- values are clipped to lie
          in the range [1/(N+2), (N+1)/(N+2)] (simply because
          non-parametric assessment lacks the power to resolve with
          higher precision in the tails, so 'imagery' samples are
          placed in each of the two tails).
        """
        self._dist_samples = np.ravel(dist_samples)
        self._correction = correction

    def __repr__(self):
        return '%s(%r%s)' % (
            self.__class__.__name__,
            self._dist_samples,
            ('', ', correction=%r' % self._correction)
              [int(self._correction != 'clip')])

    @staticmethod
    def fit(dist_samples):
        return [dist_samples]

    def _cdf(self, x, operator):
        """Helper function to compute cdf proper or reverse (i.e. going from the right tail)
        """
        res = operator(x)
        if self._correction == 'clip':
            nsamples = len(self._dist_samples)
            np.clip(res, 1.0/(nsamples+2), (nsamples+1.0)/(nsamples+2), res)
        elif self._correction is None:
            pass
        else:
            raise ValueError, \
                  '%r is incorrect value for correction parameter of %s' \
                  % (self._correction, self.__class__.__name__)
        return res


    def cdf(self, x):
        """Returns the cdf value at `x`.
        """
        return self._cdf(x,
                         np.vectorize(lambda v: (self._dist_samples <= v).mean()))

    def rcdf(self, x):
        """Returns cdf of reversed distribution (i.e. if integrating from right tail)

        Necessary for hypothesis testing in the right tail.
        It is really just a 1 - cdf(x) + pmf(x) == sf(x)+pmf(x) for a discrete distribution
        """
        return self._cdf(x,
                         np.vectorize(lambda v: (self._dist_samples >= v).mean()))


def _pvalue(x, cdf_func, rcdf_func, tail, return_tails=False, name=None):
    """Helper function to return p-value(x) given cdf and tail

    Parameters
    ----------
    cdf_func : callable
      Function to be used to derive cdf values for x
    tail : str ('left', 'right', 'any', 'both')
      Which tail of the distribution to report. For 'any' and 'both'
      it chooses the tail it belongs to based on the comparison to
      p=0.5. In the case of 'any' significance is taken like in a
      one-tailed test.
    return_tails : bool
      If True, a tuple return (pvalues, tails), where tails contain
      1s if value was from the right tail, and 0 if the value was
      from the left tail.
    """
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = [x]

    def stability_assurance(cdf):
        if __debug__ and 'CHECK_STABILITY' in debug.active:
            cdf_min, cdf_max = np.min(cdf), np.max(cdf)
            if cdf_min < 0 or cdf_max > 1.0:
                s = ('', ' for %s' % name)[int(name is not None)]
                warning('Stability check of cdf %s failed%s. Min=%s, max=%s' % \
                        (cdf_func, s, cdf_min, cdf_max))

    if tail == 'left':
        pvalues = cdf_func(x)
        if return_tails:
            right_tail = np.zeros(pvalues.shape, dtype=bool)
        stability_assurance(pvalues)
    elif tail == 'right':
        pvalues = rcdf_func(x)
        if return_tails:
            right_tail = np.ones(pvalues.shape, dtype=bool)
        stability_assurance(pvalues)
    elif tail in ('any', 'both'):
        pvalues = cdf_func(x)
        right_tail = (pvalues >= 0.5)

        if np.any(right_tail):
            # we must compute them all first ATM since otherwise
            # it would not work for "multiple" features with independent
            # distributions
            rcdf = rcdf_func(x)
            # and then assign the "interesting" ones
            pvalues[right_tail] = rcdf[right_tail]
        if tail == 'both':
            # we need report the area under both tails
            # XXX this is only meaningful for symmetric distributions
            pvalues *= 2

    # no escape but to assure that CDF is in the right range. Some
    # distributions from scipy tend to jump away from [0,1]
    # yoh: made inplace operation whenever RF into this function
    np.clip(pvalues, 0, 1.0, pvalues)

    # Assure that NaNs didn't get significant value
    # TODO: should be moved into corresponding cdf/rcdf computation
    #       since that is where x->pvalues relation can be assured
    x_nans = np.isnan(x)
    if np.any(x_nans):
        if (isinstance(x, np.ndarray) and x.shape == pvalues.shape) \
          or (pvalues.ndim == 1 and len(x) == len(pvalues)):
            pvalues[x_nans] = 1.0
        else:
            raise ValueError(
                "Input had NaN's but of different shape %s than output "
                "pvalues %s, so cannot deduce what needs to be done. Please "
                "make your input cleaner" % (x.shape, pvalues.shape))

    if is_scalar:
        pvalues = pvalues[0]

    if return_tails:
        return (pvalues, right_tail)
    else:
        return pvalues


class NullDist(ClassWithCollections):
    """Base class for null-hypothesis testing.

    """

    # Although base class is not benefiting from ca, derived
    # classes do (MCNullDist). For the sake of avoiding multiple
    # inheritance and associated headache -- let them all be ClassWithCollections,
    # performance hit should be negligible in most of the scenarios
    _ATTRIBUTE_COLLECTIONS = ['ca']

    def __init__(self, tail='both', **kwargs):
        """
        Parameters
        ----------
        tail : {'left', 'right', 'any', 'both'}
          Which tail of the distribution to report. For 'any' and 'both'
          it chooses the tail it belongs to based on the comparison to
          p=0.5. In the case of 'any' significance is taken like in a
          one-tailed test.
        """
        ClassWithCollections.__init__(self, **kwargs)

        self._set_tail(tail)

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(NullDist, self).__repr__(
            prefixes=["tail=%s" % `self.__tail`] + prefixes)


    ##REF: Name was automagically refactored
    def _set_tail(self, tail):
        # sanity check
        if tail not in ('left', 'right', 'any', 'both'):
            raise ValueError, 'Unknown value "%s" to `tail` argument.' \
                  % tail
        self.__tail = tail


    def fit(self, measure, ds):
        """Implement to fit the distribution to the data."""
        raise NotImplementedError


    def cdf(self, x):
        """Implementations return the value of the cumulative distribution
        function.
        """
        raise NotImplementedError

    def rcdf(self, x):
        """Implementations return the value of the reverse cumulative distribution
        function.
        """
        raise NotImplementedError

    def dists(self):
        """Implementations returns a sequence of the ``dist_class`` instances
        that were used to fit the distribution.
        """
        raise NotImplementedError

    def p(self, x, return_tails=False, **kwargs):
        """Returns the p-value for values of `x`.
        Returned values are determined left, right, or from any tail
        depending on the constructor setting.

        In case a `FeaturewiseMeasure` was used to estimate the
        distribution the method returns an array. In that case `x` can be
        a scalar value or an array of a matching shape.
        """
        peas = _pvalue(x, self.cdf, self.rcdf, self.__tail, return_tails=return_tails,
                       **kwargs)
        if is_datasetlike(x):
            # return the p-values in a dataset as well and assign the input
            # dataset attributes to the return dataset too
            pds = x.copy(deep=False)
            if return_tails:
                pds.samples = peas[0]
                return pds, peas[1]
            else:
                pds.samples = peas
                return pds
        return peas

    tail = property(fget=lambda x:x.__tail, fset=_set_tail)


class MCNullDist(NullDist):
    """Null-hypothesis distribution is estimated from randomly permuted data labels.

    The distribution is estimated by calling fit() with an appropriate
    `Measure` or `TransferError` instance and a training and a
    validation dataset (in case of a `TransferError`). For a customizable
    amount of cycles the training data labels are permuted and the
    corresponding measure computed. In case of a `TransferError` this is the
    error when predicting the *correct* labels of the validation dataset.

    The distribution can be queried using the `cdf()` method, which can be
    configured to report probabilities/frequencies from `left` or `right` tail,
    i.e. fraction of the distribution that is lower or larger than some
    critical value.

    This class also supports `FeaturewiseMeasure`. In that case `cdf()`
    returns an array of featurewise probabilities/frequencies.
    """

    _DEV_DOC = """
    TODO automagically decide on the number of samples/permutations needed
    Caution should be paid though since resultant distributions might be
    quite far from some conventional ones (e.g. Normal) -- it is expected to
    them to be bimodal (or actually multimodal) in many scenarios.
    """

    dist_samples = ConditionalAttribute(enabled=False,
                                 doc='Samples obtained for each permutation')
    skipped = ConditionalAttribute(enabled=True,
                  doc='# of the samples which were skipped because '
                      'measure has failed to evaluated at them')

    def __init__(self, permutator, dist_class=Nonparametric, measure=None,
                 **kwargs):
        """Initialize Monte-Carlo Permutation Null-hypothesis testing

        Parameters
        ----------
        permutator : Node
          Node instance that generates permuted datasets.
        dist_class : class
          This can be any class which provides parameters estimate
          using `fit()` method to initialize the instance, and
          provides `cdf(x)` method for estimating value of x in CDF.
          All distributions from SciPy's 'stats' module can be used.
        measure : Measure or None
          Optional measure that is used to compute results on permuted
          data. If None, a measure needs to be passed to ``fit()``.
        """
        NullDist.__init__(self, **kwargs)

        self._dist_class = dist_class
        self._dist = []                 # actual distributions
        self._measure = measure

        self.__permutator = permutator

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        prefixes_ = ["%s" % self.__permutator]
        if self._dist_class != Nonparametric:
            prefixes_.insert(0, 'dist_class=%r' % (self._dist_class,))
        return super(MCNullDist, self).__repr__(
            prefixes=prefixes_ + prefixes)


    def fit(self, measure, ds):
        """Fit the distribution by performing multiple cycles which repeatedly
        permuted labels in the training dataset.

        Parameters
        ----------
        measure: Measure or None
          A measure used to compute the results from shuffled data. Can be None
          if a measure instance has been provided to the constructor.
        ds: `Dataset` which gets permuted and used to compute the
          measure/transfer error multiple times.
        """
        # TODO: place exceptions separately so we could avoid circular imports
        from mvpa2.base.learner import LearnerError

        # prefer the already assigned measure over anything the was passed to
        # the function.
        # XXX that is a bit awkward but is necessary to keep the code changes
        # in the rest of PyMVPA minimal till this behavior become mandatory
        if self._measure is not None:
            measure = self._measure
            measure.untrain()

        dist_samples = []
        """Holds the values for randomized labels."""

        # estimate null-distribution
        # TODO this really needs to be more clever! If data samples are
        # shuffled within a class it really makes no difference for the
        # classifier, hence the number of permutations to estimate the
        # null-distribution of transfer errors can be reduced dramatically
        # when the *right* permutations (the ones that matter) are done.
        skipped = 0                     # # of skipped permutations
        for p, permuted_ds in enumerate(self.__permutator.generate(ds)):
            # new permutation all the time
            # but only permute the training data and keep the testdata constant
            #
            if __debug__:
                debug('STATMC', "Doing %i permutations: %i" \
                      % (self.__permutator.count, p+1), cr=True)

            # compute and store the measure of this permutation
            # assume it has `TransferError` interface
            try:
                res = measure(permuted_ds)
                dist_samples.append(res.samples)
            except LearnerError, e:
                if __debug__:
                    debug('STATMC', " skipped", cr=True)
                warning('Failed to obtain value from %s due to %s.  Measurement'
                        ' was skipped, which could lead to unstable and/or'
                        ' incorrect assessment of the null_dist' % (measure, e))
                skipped += 1
                continue

        self.ca.skipped = skipped

        if __debug__:
            debug('STATMC', ' Skipped: %d permutations' % skipped)

        if not len(dist_samples) and skipped > 0:
            raise RuntimeError(
                'Failed to obtain any value from %s. %d measurements were '
                'skipped. Check above warnings, and your code/data'
                % (measure, skipped))
        # store samples as (npermutations x nsamples x nfeatures)
        dist_samples = np.asanyarray(dist_samples)
        # for the ca storage use a dataset with
        # (nsamples x nfeatures x npermutations) to make it compatible with the
        # result dataset of the measure
        self.ca.dist_samples = Dataset(np.rollaxis(dist_samples,
                                       0, len(dist_samples.shape)))

        # fit distribution per each element

        # to decide either it was done on scalars or vectors
        shape = dist_samples.shape
        nshape = len(shape)
        # if just 1 dim, original data was scalar, just create an
        # artif dimension for it
        if nshape == 1:
            dist_samples = dist_samples[:, np.newaxis]

        # fit per each element.
        # XXX could be more elegant? may be use np.vectorize?
        dist_samples_rs = dist_samples.reshape((shape[0], -1))
        dist = []
        for samples in dist_samples_rs.T:
            params = self._dist_class.fit(samples)
            if __debug__ and 'STAT__' in debug.active:
                debug('STAT', 'Estimated parameters for the %s are %s'
                      % (self._dist_class, str(params)))
            dist.append(self._dist_class(*params))
        self._dist = dist


    def _cdf(self, x, cdf_func):
        """Return value of the cumulative distribution function at `x`.
        """
        if self._dist is None:
            # XXX We might not want to descriminate that way since
            # usually generators also have .cdf where they rely on the
            # default parameters. But then what about Nonparametric
            raise RuntimeError, "Distribution has to be fit first"

        is_scalar = np.isscalar(x)
        if is_scalar:
            x = [x]

        x = np.asanyarray(x)
        xshape = x.shape
        # assure x is a 1D array now
        x = x.reshape((-1,))

        if len(self._dist) != len(x):
            raise ValueError, 'Distribution was fit for structure with %d' \
                  ' elements, whenever now queried with %d elements' \
                  % (len(self._dist), len(x))

        # extract cdf values per each element
        if cdf_func == 'cdf':
            cdfs = [ dist.cdf(v) for v, dist in zip(x, self._dist) ]
        elif cdf_func == 'rcdf':
            cdfs = [ _auto_rcdf(dist)(v) for v, dist in zip(x, self._dist) ]
        else:
            raise ValueError
        return np.array(cdfs).reshape(xshape)

    def cdf(self, x):
        return self._cdf(x, 'cdf')

    def rcdf(self, x):
        return self._cdf(x, 'rcdf')

    def dists(self):
        return self._dist

    def clean(self):
        """Clean stored distributions

        Storing all of the distributions might be too expensive
        (e.g. in case of Nonparametric), and the scope of the object
        might be too broad to wait for it to be destroyed. Clean would
        bind dist_samples to empty list to let gc revoke the memory.
        """
        self._dist = []



class FixedNullDist(NullDist):
    """Proxy/Adaptor class for SciPy distributions.

    All distributions from SciPy's 'stats' module can be used with this class.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy import stats
    >>> from mvpa2.clfs.stats import FixedNullDist
    >>>
    >>> dist = FixedNullDist(stats.norm(loc=2, scale=4), tail='left')
    >>> dist.p(2)
    0.5
    >>>
    >>> dist.cdf(np.arange(5))
    array([ 0.30853754,  0.40129367,  0.5       ,  0.59870633,  0.69146246])
    >>>
    >>> dist = FixedNullDist(stats.norm(loc=2, scale=4), tail='right')
    >>> dist.p(np.arange(5))
    array([ 0.69146246,  0.59870633,  0.5       ,  0.40129367,  0.30853754])

    """
    def __init__(self, dist, **kwargs):
        """
        Parameters
        ----------
        dist : distribution object
          This can be any object the has a `cdf()` method to report the
          cumulative distribition function values.
        """
        NullDist.__init__(self, **kwargs)

        self._dist = dist
        # assign corresponding rcdf overloading NotImplemented one of
        # base class
        self.rcdf = _auto_rcdf(dist)

    def fit(self, measure, ds):
        """Does nothing since the distribution is already fixed."""
        pass


    def cdf(self, x):
        """Return value of the cumulative distribution function at `x`.
        """
        return self._dist.cdf(x)


    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        prefixes_ = ["dist=%s" % `self._dist`]
        return super(FixedNullDist, self).__repr__(
            prefixes=prefixes_ + prefixes)


class AdaptiveNullDist(FixedNullDist):
    """Adaptive distribution which adjusts parameters according to the data

    WiP: internal implementation might change
    """
    def fit(self, measure, wdata, vdata=None):
        """Cares about dimensionality of the feature space in measure
        """

        try:
            nfeatures = len(measure.feature_ids)
        except ValueError:              # XXX
            nfeatures = np.prod(wdata.shape[1:])

        dist_gen = self._dist
        if not hasattr(dist_gen, 'fit'): # frozen already
            dist_gen = dist_gen.dist     # rv_frozen at least has it ;)

        args, kwargs = self._adapt(nfeatures, measure, wdata, vdata)
        if __debug__:
            debug('STAT', 'Adapted parameters for %s to be %s, %s'
                  % (dist_gen, args, kwargs))
        self._dist = dist_gen(*args, **kwargs)


    def _adapt(self, nfeatures, measure, wdata, vdata=None):
        raise NotImplementedError


class AdaptiveRDist(AdaptiveNullDist):
    """Adaptive rdist: params are (nfeatures-1, 0, 1)
    """

    def _adapt(self, nfeatures, measure, wdata, vdata=None):
        return (nfeatures-1, 0, 1), {}

    # XXX: RDist has stability issue, just run
    #  python -c "import scipy.stats; print scipy.stats.rdist(541,0,1).cdf(0.72)"
    # to get some improbable value, so we need to take care about that manually
    # here
    def cdf(self, x):
        cdf_ = self._dist.cdf(x)
        bad_values = np.where(np.abs(cdf_)>1)
        # XXX there might be better implementation (faster/elegant) using np.clip,
        #     the only problem is that instability results might flip the sign
        #     arbitrarily
        if len(bad_values[0]):
            # in this distribution we have mean at 0, so we can take that easily
            # into account
            cdf_bad = cdf_[bad_values]
            x_bad = x[bad_values]
            cdf_bad[x_bad < 0] = 0.0
            cdf_bad[x_bad >= 0] = 1.0
            cdf_[bad_values] = cdf_bad
        return cdf_


class AdaptiveNormal(AdaptiveNullDist):
    """Adaptive Normal Distribution: params are (0, sqrt(1/nfeatures))
    """

    def _adapt(self, nfeatures, measure, wdata, vdata=None):
        return (0, 1.0/np.sqrt(nfeatures)), {}


if externals.exists('scipy'):
    from mvpa2.support.scipy.stats import scipy
    from scipy.stats import kstest

    """
    Thoughts:

    So we can use `scipy.stats.kstest` (Kolmogorov-Smirnov test) to
    check/reject H0 that samples come from a given distribution. But
    since it is based on a full range of data, we might better of with
    some ad-hoc checking by the detection power of the values in the
    tail of a tentative distribution.

    """

    # We need a way to fixate estimation of some parameters
    # (e.g. mean) so lets create a simple proxy, or may be class
    # generator later on, which would take care about punishing change
    # from the 'right' arguments

    import scipy

    class rv_semifrozen(object):
        """Helper proxy-class to fit distribution when some parameters are known

        It is an ugly hack with snippets of code taken from scipy, which is
        Copyright (c) 2001, 2002 Enthought, Inc.
        and is distributed under BSD license
        http://www.scipy.org/License_Compatibility
        """

        def __init__(self, dist, loc=None, scale=None, args=None):
            """
            Parameters
            ----------
            dist : rv_generic
              Distribution for which to freeze some of the parameters
            loc : array-like, optional
              Location parameter (default=0)
            scale : array-like, optional
              Scale parameter (default=1)
            args : iterable, optional
               Additional arguments to be passed to dist.

            Raises
            ------
            ValueError
              Arguments number mismatch
            """
            self._dist = dist
            # loc and scale
            theta = (loc, scale)
            # args
            Narg_ = dist.numargs
            if args is not None:
                Narg = len(args)
                if Narg > Narg_:
                    raise ValueError, \
                          'Distribution %s has only %d arguments. Got %d' \
                          % (dist, Narg_, Narg)
                args += (None,) * (Narg_ - Narg)
            else:
                args = (None,) * Narg_

            args_i = [i for i,v in enumerate(args) if v is None]
            self._fargs = (list(args+theta), args_i)
            """Arguments which should get some fixed value"""


        def __call__(self, *args, **kwargs):
            """Upon call mimic call to get actual rv_frozen distribution
            """
            return self._dist(*args, **kwargs)


        def nnlf(self, theta, x):
            # - sum (log pdf(x, theta),axis=0)
            #   where theta are the parameters (including loc and scale)
            #
            fargs, fargs_i = self._fargs
            try:
                i=-1
                if fargs[-1] is not None:
                    scale = fargs[-1]
                else:
                    scale = theta[i]
                    i -= 1

                if fargs[-2] is not None:
                    loc = fargs[-2]
                else:
                    loc = theta[i]
                    i -= 1

                args = theta[:i+1]
                # adjust args if there were fixed
                for i, a in zip(fargs_i, args):
                    fargs[i] = a
                args = fargs[:-2]

            except IndexError:
                raise ValueError, "Not enough input arguments."
            if not self._argcheck(*args) or scale <= 0:
                return np.inf
            x = np.asarray((x-loc) / scale)
            cond0 = (x <= self.a) | (x >= self.b)
            if (np.any(cond0)):
                return np.inf
            else:
                return self._nnlf(x, *args) + len(x)*np.log(scale)

        def fit(self, data, *args, **kwds):
            loc0, scale0 = map(kwds.get, ['loc', 'scale'], [0.0, 1.0])
            fargs, fargs_i = self._fargs
            Narg = len(args)
            Narg_ = self.numargs
            if Narg != Narg_:
                if Narg > Narg_:
                    raise ValueError, "Too many input arguments."
                else:
                    args += (1.0,)*(self.numargs-Narg)

            # Provide only those args which are not fixed, and
            # append location and scale (if not fixed) at the end
            if len(fargs_i) != Narg_:
                x0 = tuple([args[i] for i in fargs_i])
            else:
                x0 = args
            if fargs[-2] is None:
                x0 = x0 + (loc0,)
            if fargs[-1] is None:
                x0 = x0 + (scale0,)

            opt_x = scipy.optimize.fmin(
                self.nnlf, x0, args=(np.ravel(data),), disp=0)

            # reconstruct back
            i = 0
            loc, scale = fargs[-2:]
            if fargs[-1] is None:
                i -= 1
                scale = opt_x[i]
            if fargs[-2] is None:
                i -= 1
                loc = opt_x[i]

            # assign those which weren't fixed
            for i in fargs_i:
                fargs[i] = opt_x[i]

            #raise ValueError
            opt_x = np.hstack((fargs[:-2], (loc, scale)))
            return opt_x


        def __setattr__(self, a, v):
            if not a in ['_dist', '_fargs', 'fit', 'nnlf']:
                self._dist.__setattr__(a, v)
            else:
                object.__setattr__(self, a, v)


        def __getattribute__(self, a):
            """We need to redirect all queries correspondingly
            """
            if not a in ['_dist', '_fargs', 'fit', 'nnlf']:
                return getattr(self._dist, a)
            else:
                return object.__getattribute__(self, a)



    ##REF: Name was automagically refactored
    def match_distribution(data, nsamples=None, loc=None, scale=None,
                          args=None, test='kstest', distributions=None,
                          **kwargs):
        """Determine best matching distribution.

        Can be used for 'smelling' the data, as well to choose a
        parametric distribution for data obtained from non-parametric
        testing (e.g. `MCNullDist`).

        WiP: use with caution, API might change

        Parameters
        ----------
        data : np.ndarray
          Array of the data for which to deduce the distribution. It has
          to be sufficiently large to make a reliable conclusion
        nsamples : int or None
          If None -- use all samples in data to estimate parametric
          distribution. Otherwise use only specified number randomly selected
          from data.
        loc : float or None
          Loc for the distribution (if known)
        scale : float or None
          Scale for the distribution (if known)
        test : str
          What kind of testing to do. Choices:
           'p-roc'
             detection power for a given ROC. Needs two
             parameters: `p=0.05` and `tail='both'`
           'kstest'
             'full-body' distribution comparison. The best
             choice is made by minimal reported distance after estimating
             parameters of the distribution. Parameter `p=0.05` sets
             threshold to reject null-hypothesis that distribution is the
             same.
             **WARNING:** older versions (e.g. 0.5.2 in etch) of scipy have
             incorrect kstest implementation and do not function properly.
        distributions : None or list of str or tuple(str, dict)
          Distributions to check. If None, all known in scipy.stats
          are tested. If distribution is specified as a tuple, then
          it must contain name and additional parameters (name, loc,
          scale, args) in the dictionary. Entry 'scipy' adds all known
          in scipy.stats.
        **kwargs
          Additional arguments which are needed for each particular test
          (see above)

        Examples
        --------
        >>> from mvpa2.clfs.stats import match_distribution
        >>> data = np.random.normal(size=(1000,1));
        >>> matches = match_distribution(
        ...   data,
        ...   distributions=['rdist',
        ...                  ('rdist', {'name':'rdist_fixed',
        ...                             'loc': 0.0,
        ...                             'args': (10,)})],
        ...   nsamples=30, test='p-roc', p=0.05)

        """

        # Handle parameters
        _KNOWN_TESTS = ['p-roc', 'kstest']
        if not test in _KNOWN_TESTS:
            raise ValueError, 'Unknown kind of test %s. Known are %s' \
                  % (test, _KNOWN_TESTS)

        data = np.ravel(data)
        # data sampled
        if nsamples is not None:
            if __debug__:
                debug('STAT', 'Sampling %d samples from data for the ' \
                      'estimation of the distributions parameters' % nsamples)
            indexes_selected = (np.random.sample(nsamples)*len(data)).astype(int)
            data_selected = data[indexes_selected]
        else:
            indexes_selected = np.arange(len(data))
            data_selected = data

        p_thr = kwargs.get('p', 0.05)
        if test == 'p-roc':
            tail = kwargs.get('tail', 'both')
            npd = Nonparametric(data)
            data_p = _pvalue(data, npd.cdf, npd.rcdf, tail)
            data_p_thr = np.abs(data_p) <= p_thr
            true_positives = np.sum(data_p_thr)
            if true_positives == 0:
                raise ValueError, "Provided data has no elements in non-" \
                      "parametric distribution under p<=%.3f. Please " \
                      "increase the size of data or value of p" % p_thr
            if __debug__:
                debug('STAT_', 'Number of positives in non-parametric '
                      'distribution is %d' % true_positives)

        if distributions is None:
            distributions = ['scipy']

        # lets see if 'scipy' entry was in there
        try:
            scipy_ind = distributions.index('scipy')
            distributions.pop(scipy_ind)
            sp_dists = ssd.__all__
            sp_version = externals.versions['scipy']
            if sp_version >= '0.9.0':
                for d_ in ['ncf']:
                    if d_ in sp_dists:
                        warning("Not considering %s distribution because of "
                                "known issues in scipy %s" % (d_, sp_version))
                        _ = sp_dists.pop(sp_dists.index(d_))
            distributions += sp_dists
        except ValueError:
            pass

        results = []
        for d in distributions:
            dist_gen, loc_, scale_, args_ = None, loc, scale, args
            if isinstance(d, basestring):
                dist_gen = d
                dist_name = d
            elif isinstance(d, tuple):
                if not (len(d)==2 and isinstance(d[1], dict)):
                    raise ValueError,\
                          "Tuple specification of distribution must be " \
                          "(d, {params}). Got %s" % (d,)
                dist_gen = d[0]
                loc_ = d[1].get('loc', loc)
                scale_ = d[1].get('scale', scale)
                args_ = d[1].get('args', args)
                dist_name = d[1].get('name', str(dist_gen))
            else:
                dist_gen = d
                dist_name = str(d)

            # perform actions which might puke for some distributions
            try:
                dist_gen_ = getattr(scipy.stats, dist_gen)
                # specify distribution 'optimizer'. If loc or scale was provided,
                # use home-brewed rv_semifrozen
                if args_ is not None or loc_ is not None or scale_ is not None:
                    dist_opt = rv_semifrozen(dist_gen_,
                                             loc=loc_, scale=scale_, args=args_)
                else:
                    dist_opt = dist_gen_

                if __debug__:
                    debug('STAT__',
                          'Fitting %s distribution %r on data of size %s',
                          (dist_name, dist_opt, data_selected.shape))
                # suppress the warnings which might pop up while
                # matching "inappropriate" distributions
                with warnings.catch_warnings(record=True) as w:
                    dist_params = dist_opt.fit(data_selected)
                if __debug__:
                    debug('STAT__',
                          'Got distribution parameters %s for %s'
                          % (dist_params, dist_name))
                if test == 'p-roc':
                    cdf_func = lambda x: dist_gen_.cdf(x, *dist_params)
                    rcdf_func = _auto_rcdf(dist_gen_)
                    # We need to compare detection under given p
                    cdf_p = np.abs(_pvalue(data, cdf_func, rcdf_func, tail, name=dist_gen))
                    cdf_p_thr = cdf_p <= p_thr
                    D, p = (np.sum(np.abs(data_p_thr - cdf_p_thr))*1.0/true_positives, 1)
                    if __debug__:
                        res_sum = 'D=%.2f' % D
                elif test == 'kstest':
                    D, p = kstest(data, dist_gen, args=dist_params)
                    if __debug__:
                        res_sum = 'D=%.3f p=%.3f' % (D, p)
            except (TypeError, ValueError, AttributeError,
                    NotImplementedError), e:#Exception, e:
                if __debug__:
                    debug('STAT__',
                          'Testing for %s distribution failed due to %s',
                          (d, e))
                continue

            if p > p_thr and not np.isnan(D):
                results += [ (D, dist_gen, dist_name, dist_params) ]
                if __debug__:
                    debug('STAT_',
                          'Tested %s distribution: %s', (dist_name, res_sum))
            else:
                if __debug__:
                    debug('STAT__', 'Cannot consider %s dist. with %s',
                          (d, res_sum))
                continue

        # sort in ascending order, so smaller is better
        results.sort(key=lambda x:x[0])

        if __debug__ and 'STAT' in debug.active:
            # find the best and report it
            nresults = len(results)
            sresult = lambda r:'%s(%s)=%.2f' % (r[1],
                                                ', '.join(map(str, r[3])),
                                                r[0])
            if nresults > 0:
                nnextbest = min(2, nresults-1)
                snextbest = ', '.join(map(sresult, results[1:1+nnextbest]))
                debug('STAT', 'Best distribution %s. Next best: %s'
                          % (sresult(results[0]), snextbest))
            else:
                debug('STAT', 'Could not find suitable distribution')

        # return all the results
        return results


    if externals.exists('pylab'):
        import pylab as pl

        ##REF: Name was automagically refactored
        def plot_distribution_matches(data, matches, nbins=31, nbest=5,
                                    expand_tails=8, legend=2, plot_cdf=True,
                                    p=None, tail='both'):
            """Plot best matching distributions

            Parameters
            ----------
            data : np.ndarray
              Data which was used to obtain the matches
            matches : list of tuples
              Sorted matches as provided by match_distribution
            nbins : int
              Number of bins in the histogram
            nbest : int
              Number of top matches to plot
            expand_tails : int
              How many bins away to add to parametrized distributions
              plots
            legend : int
              Either to provide legend and statistics in the legend.
              1 -- just lists distributions.
              2 -- adds distance measure
              3 -- tp/fp/fn in the case if p is provided
            plot_cdf : bool
              Either to plot cdf for data using non-parametric distribution
            p : float or None
              If not None, visualize null-hypothesis testing (given p).
              Bars in the histogram which fall under given p are colored
              in red. False positives and false negatives are marked as
              triangle up and down symbols correspondingly
            tail : ('left', 'right', 'any', 'both')
              If p is not None, the choise of tail for null-hypothesis
              testing

            Returns
            -------
            histogram
            list of lines
            """

            # API changed since v0.99.0-641-ga7c2231
            halign = externals.versions['matplotlib'] >= '1.0.0' \
                     and 'mid' or 'center'
            hist = pl.hist(data, nbins, normed=1, align=halign)
            data_range = [np.min(data), np.max(data)]

            # x's
            x = hist[1]
            dx = x[expand_tails] - x[0] # how much to expand tails by
            x = np.hstack((x[:expand_tails] - dx, x, x[-expand_tails:] + dx))

            nonparam = Nonparametric(data)
            # plot cdf
            if plot_cdf:
                pl.plot(x, nonparam.cdf(x), 'k--', linewidth=1)

            data_p = _pvalue(data, nonparam.cdf, nonparam.rcdf, tail)

            npd = Nonparametric(data)
            x_p = _pvalue(x, npd.cdf, npd.rcdf, tail)

            if p is not None:
                data_p_thr = (data_p <= p).ravel()
                x_p_thr = np.abs(x_p) <= p

                # color bars which pass thresholding in red
                for thr, bar_ in zip(x_p_thr[expand_tails:], hist[2]):
                    bar_.set_facecolor(('w','r')[int(thr)])

            if not len(matches):
                # no matches were provided
                warning("No matching distributions were provided -- nothing to plot")
                return (hist, )

            lines = []
            labels = []
            for i in xrange(min(nbest, len(matches))):
                D, dist_gen, dist_name, params = matches[i]
                dist = getattr(scipy.stats, dist_gen)(*params)
                rcdf = _auto_rcdf(dist)
                label = '%s' % (dist_name)
                if legend > 1:
                    label += '(D=%.2f)' % (D)

                xcdf_p = np.abs(_pvalue(x, dist.cdf, rcdf, tail))
                if p is not None:
                    xcdf_p_thr = (xcdf_p <= p).ravel()

                if p is not None and legend > 2:
                    # We need to compare detection under given p
                    data_cdf_p = np.abs(_pvalue(data, dist.cdf, rcdf, tail))
                    data_cdf_p_thr = (data_cdf_p <= p).ravel()

                    # true positives
                    tp = np.logical_and(data_cdf_p_thr, data_p_thr)
                    # false positives
                    fp = np.logical_and(data_cdf_p_thr, ~data_p_thr)
                    # false negatives
                    fn = np.logical_and(~data_cdf_p_thr, data_p_thr)

                    label += ' tp/fp/fn=%d/%d/%d)' % \
                            tuple(map(np.sum, [tp, fp, fn]))

                pdf = dist.pdf(x)
                line = pl.plot(x, pdf, '-', linewidth=2, label=label)[0]
                color = line.get_color()

                if plot_cdf:
                    cdf = dist.cdf(x)
                    pl.plot(x, cdf, ':', linewidth=1, color=color, label=label)

                # TODO: decide on tp/fp/fn by not centers of the bins but
                #       by the values in data in the ranges covered by
                #       those bins. Then it would correspond to the values
                #       mentioned in the legend
                if p is not None:
                    # true positives
                    xtp = np.logical_and(xcdf_p_thr, x_p_thr)
                    # false positives
                    xfp = np.logical_and(xcdf_p_thr, ~x_p_thr)
                    # false negatives
                    xfn = np.logical_and(~xcdf_p_thr, x_p_thr)

                    # no need to plot tp explicitely -- marked by color of the bar
                    # pl.plot(x[xtp], pdf[xtp], 'o', color=color)
                    pl.plot(x[xfp], pdf[xfp], '^', color=color)
                    pl.plot(x[xfn], pdf[xfn], 'v', color=color)

                lines.append(line)
                labels.append(label)

            if legend:
                pl.legend(lines, labels)

            return (hist, lines)

    #if True:
    #    data = np.random.normal(size=(1000,1));
    #    matches = match_distribution(
    #        data,
    #        distributions=['scipy',
    #                       ('norm', {'name':'norm_known',
    #                                 'scale': 1.0,
    #                                 'loc': 0.0})],
    #        nsamples=30, test='p-roc', p=0.05)
    #    pl.figure(); plot_distribution_matches(data, matches, nbins=101,
    #                                        p=0.05, legend=4, nbest=5)


##REF: Name was automagically refactored
def auto_null_dist(dist):
    """Cheater for human beings -- wraps `dist` if needed with some
    NullDist

    tail and other arguments are assumed to be default as in
    NullDist/MCNullDist
    """
    if dist is None or isinstance(dist, NullDist):
        return dist
    elif hasattr(dist, 'fit'):
        if __debug__:
            debug('STAT', 'Wrapping %s into MCNullDist' % dist)
        return MCNullDist(dist)
    else:
        if __debug__:
            debug('STAT', 'Wrapping %s into FixedNullDist' % dist)
        return FixedNullDist(dist)


# if no scipy, we need nanmean
def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis

def nanmean(x, axis=0):
    """Compute the mean over the given axis ignoring NaNs.

    Parameters
    ----------
    x : ndarray
      input array
    axis : int
      axis along which the mean is computed.

    Returns
    -------
    m : float
      the mean.
    """
    x, axis = _chk_asarray(x, axis)
    x = x.copy()
    Norig = x.shape[axis]
    factor = 1.0 - np.sum(np.isnan(x), axis)*1.0/Norig

    x[np.isnan(x)] = 0
    return np.mean(x, axis)/factor
