# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from mvpa.misc.state import ClassWithCollections, StateVariable

if __debug__:
    from mvpa.base import debug


class Nonparametric(object):
    """Non-parametric 1d distribution -- derives cdf based on stored values.

    Introduced to complement parametric distributions present in scipy.stats.
    """

    def __init__(self, dist_samples):
        self._dist_samples = N.ravel(dist_samples)


    @staticmethod
    def fit(dist_samples):
        return [dist_samples]


    def cdf(self, x):
        """Returns the cdf value at `x`.
        """
        return N.vectorize(lambda v:(self._dist_samples <= v).mean())(x)


def _pvalue(x, cdf_func, tail, return_tails=False, name=None):
    """Helper function to return p-value(x) given cdf and tail

    :Parameters:
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
    is_scalar = N.isscalar(x)
    if is_scalar:
        x = [x]

    cdf = cdf_func(x)

    if __debug__ and 'CHECK_STABILITY' in debug.active:
        cdf_min, cdf_max = N.min(cdf), N.max(cdf)
        if cdf_min < 0 or cdf_max > 1.0:
            s = ('', ' for %s' % name)[int(name is not None)]
            warning('Stability check of cdf %s failed%s. Min=%s, max=%s' % \
                  (cdf_func, s, cdf_min, cdf_max))

    # no escape but to assure that CDF is in the right range. Some
    # distributions from scipy tend to jump away from [0,1]
    cdf = N.clip(cdf, 0, 1.0)

    if tail == 'left':
        if return_tails:
            right_tail = N.zeros(cdf.shape, dtype=bool)
    elif tail == 'right':
        cdf = 1 - cdf
        if return_tails:
            right_tail = N.ones(cdf.shape, dtype=bool)
    elif tail in ('any', 'both'):
        right_tail = (cdf >= 0.5)
        cdf[right_tail] = 1.0 - cdf[right_tail]
        if tail == 'both':
            # we need to half the signficance
            cdf *= 2

    # Assure that NaNs didn't get significant value
    cdf[N.isnan(x)] = 1.0
    if is_scalar: res = cdf[0]
    else:         res = cdf

    if return_tails:
        return (res, right_tail)
    else:
        return res


class NullDist(ClassWithCollections):
    """Base class for null-hypothesis testing.

    """

    # Although base class is not benefiting from states, derived
    # classes do (MCNullDist). For the sake of avoiding multiple
    # inheritance and associated headache -- let them all be ClassWithCollections,
    # performance hit should be negligible in most of the scenarios
    _ATTRIBUTE_COLLECTIONS = ['states']

    def __init__(self, tail='both', **kwargs):
        """Cheap initialization.

        :Parameter:
          tail: str ('left', 'right', 'any', 'both')
            Which tail of the distribution to report. For 'any' and 'both'
            it chooses the tail it belongs to based on the comparison to
            p=0.5. In the case of 'any' significance is taken like in a
            one-tailed test.
        """
        ClassWithCollections.__init__(self, **kwargs)

        self._setTail(tail)


    def _setTail(self, tail):
        # sanity check
        if tail not in ('left', 'right', 'any', 'both'):
            raise ValueError, 'Unknown value "%s" to `tail` argument.' \
                  % tail
        self.__tail = tail


    def fit(self, measure, wdata, vdata=None):
        """Implement to fit the distribution to the data."""
        raise NotImplementedError


    def cdf(self, x):
        """Implementations return the value of the cumulative distribution
        function (left or right tail dpending on the setting).
        """
        raise NotImplementedError


    def p(self, x, **kwargs):
        """Returns the p-value for values of `x`.
        Returned values are determined left, right, or from any tail
        depending on the constructor setting.

        In case a `FeaturewiseDatasetMeasure` was used to estimate the
        distribution the method returns an array. In that case `x` can be
        a scalar value or an array of a matching shape.
        """
        return _pvalue(x, self.cdf, self.__tail, **kwargs)


    tail = property(fget=lambda x:x.__tail, fset=_setTail)


class MCNullDist(NullDist):
    """Null-hypothesis distribution is estimated from randomly permuted data labels.

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
    them to be bimodal (or actually multimodal) in many scenarios.
    """

    dist_samples = StateVariable(enabled=False,
                                 doc='Samples obtained for each permutation')

    def __init__(self, dist_class=Nonparametric, permutations=100, **kwargs):
        """Initialize Monte-Carlo Permutation Null-hypothesis testing

        :Parameters:
          dist_class: class
            This can be any class which provides parameters estimate
            using `fit()` method to initialize the instance, and
            provides `cdf(x)` method for estimating value of x in CDF.
            All distributions from SciPy's 'stats' module can be used.
          permutations: int
            This many permutations of label will be performed to
            determine the distribution under the null hypothesis.
        """
        NullDist.__init__(self, **kwargs)

        self._dist_class = dist_class
        self._dist = []                 # actual distributions

        self.__permutations = permutations
        """Number of permutations to compute the estimate the null
        distribution."""



    def fit(self, measure, wdata, vdata=None):
        """Fit the distribution by performing multiple cycles which repeatedly
        permuted labels in the training dataset.

        :Parameters:
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

        # fit per each element.
        # XXX could be more elegant? may be use N.vectorize?
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

        x = N.asanyarray(x)
        xshape = x.shape
        # assure x is a 1D array now
        x = x.reshape((-1,))

        if len(self._dist) != len(x):
            raise ValueError, 'Distribution was fit for structure with %d' \
                  ' elements, whenever now queried with %d elements' \
                  % (len(self._dist), len(x))

        # extract cdf values per each element
        cdfs = [ dist.cdf(v) for v, dist in zip(x, self._dist) ]
        return N.array(cdfs).reshape(xshape)


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

    >>> import numpy as N
    >>> from scipy import stats
    >>> from mvpa.clfs.stats import FixedNullDist
    >>>
    >>> dist = FixedNullDist(stats.norm(loc=2, scale=4))
    >>> dist.p(2)
    0.5
    >>>
    >>> dist.cdf(N.arange(5))
    array([ 0.30853754,  0.40129367,  0.5       ,  0.59870633,  0.69146246])
    >>>
    >>> dist = FixedNullDist(stats.norm(loc=2, scale=4), tail='right')
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
        NullDist.__init__(self, **kwargs)

        self._dist = dist


    def fit(self, measure, wdata, vdata=None):
        """Does nothing since the distribution is already fixed."""
        pass


    def cdf(self, x):
        """Return value of the cumulative distribution function at `x`.
        """
        return self._dist.cdf(x)


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
            nfeatures = N.prod(wdata.shape[1:])

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
        bad_values = N.where(N.abs(cdf_)>1)
        # XXX there might be better implementation (faster/elegant) using N.clip,
        #     the only problem is that instability results might flip the sign
        #     arbitrarily
        if len(bad_values[0]):
            # in this distribution we have mean at 0, so we can take that easily
            # into account
            cdf_bad = cdf_[bad_values]
            x_bad = x[bad_values]
            cdf_bad[x_bad<0] = 0.0
            cdf_bad[x_bad>=0] = 1.0
            cdf_[bad_values] = cdf_bad
        return cdf_


class AdaptiveNormal(AdaptiveNullDist):
    """Adaptive Normal Distribution: params are (0, sqrt(1/nfeatures))
    """

    def _adapt(self, nfeatures, measure, wdata, vdata=None):
        return (0, 1.0/N.sqrt(nfeatures)), {}


if externals.exists('scipy'):
    from mvpa.support.stats import scipy
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
                for i,a in zip(fargs_i, args):
                    fargs[i] = a
                args = fargs[:-2]

            except IndexError:
                raise ValueError, "Not enough input arguments."
            if not self._argcheck(*args) or scale <= 0:
                return N.inf
            x = N.asarray((x-loc) / scale)
            cond0 = (x <= self.a) | (x >= self.b)
            if (N.any(cond0)):
                return N.inf
            else:
                return self._nnlf(x, *args) + len(x)*N.log(scale)

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
            if fargs[-2] is None: x0 = x0 + (loc0,)
            if fargs[-1] is None: x0 = x0 + (scale0,)

            opt_x = scipy.optimize.fmin(self.nnlf, x0, args=(N.ravel(data),), disp=0)

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
            opt_x = N.hstack((fargs[:-2], (loc, scale)))
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



    def matchDistribution(data, nsamples=None, loc=None, scale=None,
                          args=None, test='kstest', distributions=None,
                          **kwargs):
        """Determine best matching distribution.

        Can be used for 'smelling' the data, as well to choose a
        parametric distribution for data obtained from non-parametric
        testing (e.g. `MCNullDist`).

        WiP: use with caution, API might change

        :Parameters:
          data : N.ndarray
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
          test : basestring
            What kind of testing to do. Choices:
             'p-roc' : detection power for a given ROC. Needs two
               parameters: `p=0.05` and `tail='both'`
             'kstest' : 'full-body' distribution comparison. The best
               choice is made by minimal reported distance after estimating
               parameters of the distribution. Parameter `p=0.05` sets
               threshold to reject null-hypothesis that distribution is the
               same.
               WARNING: older versions (e.g. 0.5.2 in etch) of scipy have
                        incorrect kstest implementation and do not function
                        properly
          distributions : None or list of basestring or tuple(basestring, dict)
            Distributions to check. If None, all known in scipy.stats
            are tested. If distribution is specified as a tuple, then
            it must contain name and additional parameters (name, loc,
            scale, args) in the dictionary. Entry 'scipy' adds all known
            in scipy.stats.
          **kwargs
            Additional arguments which are needed for each particular test
            (see above)

        :Example:
          data = N.random.normal(size=(1000,1));
          matches = matchDistribution(
            data,
            distributions=['rdist',
                           ('rdist', {'name':'rdist_fixed',
                                      'loc': 0.0,
                                      'args': (10,)})],
            nsamples=30, test='p-roc', p=0.05)
        """

        # Handle parameters
        _KNOWN_TESTS = ['p-roc', 'kstest']
        if not test in _KNOWN_TESTS:
            raise ValueError, 'Unknown kind of test %s. Known are %s' \
                  % (test, _KNOWN_TESTS)

        data = N.ravel(data)
        # data sampled
        if nsamples is not None:
            if __debug__:
                debug('STAT', 'Sampling %d samples from data for the ' \
                      'estimation of the distributions parameters' % nsamples)
            indexes_selected = (N.random.sample(nsamples)*len(data)).astype(int)
            data_selected = data[indexes_selected]
        else:
            indexes_selected = N.arange(len(data))
            data_selected = data

        p_thr = kwargs.get('p', 0.05)
        if test == 'p-roc':
            tail = kwargs.get('tail', 'both')
            data_p = _pvalue(data, Nonparametric(data).cdf, tail)
            data_p_thr = N.abs(data_p) <= p_thr
            true_positives = N.sum(data_p_thr)
            if true_positives == 0:
                raise ValueError, "Provided data has no elements in non-" \
                      "parametric distribution under p<=%.3f. Please " \
                      "increase the size of data or value of p" % p
            if __debug__:
                debug('STAT_', 'Number of positives in non-parametric '
                      'distribution is %d' % true_positives)

        if distributions is None:
            distributions = ['scipy']

        # lets see if 'scipy' entry was in there
        try:
            scipy_ind = distributions.index('scipy')
            distributions.pop(scipy_ind)
            distributions += scipy.stats.distributions.__all__
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
                    dist_opt = rv_semifrozen(dist_gen_, loc=loc_, scale=scale_, args=args_)
                else:
                    dist_opt = dist_gen_
                dist_params = dist_opt.fit(data_selected)
                if __debug__:
                    debug('STAT__',
                          'Got distribution parameters %s for %s'
                          % (dist_params, dist_name))
                if test == 'p-roc':
                    cdf_func = lambda x: dist_gen_.cdf(x, *dist_params)
                    # We need to compare detection under given p
                    cdf_p = N.abs(_pvalue(data, cdf_func, tail, name=dist_gen))
                    cdf_p_thr = cdf_p <= p_thr
                    D, p = N.sum(N.abs(data_p_thr - cdf_p_thr))*1.0/true_positives, 1
                    if __debug__: res_sum = 'D=%.2f' % D
                elif test == 'kstest':
                    D, p = kstest(data, d, args=dist_params)
                    if __debug__: res_sum = 'D=%.3f p=%.3f' % (D, p)
            except (TypeError, ValueError, AttributeError,
                    NotImplementedError), e:#Exception, e:
                if __debug__:
                    debug('STAT__',
                          'Testing for %s distribution failed due to %s'
                          % (d, str(e)))
                continue

            if p > p_thr and not N.isnan(D):
                results += [ (D, dist_gen, dist_name, dist_params) ]
                if __debug__:
                    debug('STAT_', 'Testing for %s dist.: %s' % (dist_name, res_sum))
            else:
                if __debug__:
                    debug('STAT__', 'Cannot consider %s dist. with %s'
                          % (d, res_sum))
                continue

        # sort in ascending order, so smaller is better
        results.sort()

        if __debug__ and 'STAT' in debug.active:
            # find the best and report it
            nresults = len(results)
            sresult = lambda r:'%s(%s)=%.2f' % (r[1], ', '.join(map(str, r[3])), r[0])
            if nresults>0:
                nnextbest = min(2, nresults-1)
                snextbest = ', '.join(map(sresult, results[1:1+nnextbest]))
                debug('STAT', 'Best distribution %s. Next best: %s'
                          % (sresult(results[0]), snextbest))
            else:
                debug('STAT', 'Could not find suitable distribution')

        # return all the results
        return results


    if externals.exists('pylab'):
        import pylab as P

        def plotDistributionMatches(data, matches, nbins=31, nbest=5,
                                    expand_tails=8, legend=2, plot_cdf=True,
                                    p=None, tail='both'):
            """Plot best matching distributions

            :Parameters:
              data : N.ndarray
                Data which was used to obtain the matches
              matches : list of tuples
                Sorted matches as provided by matchDistribution
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

            :Returns: tuple(histogram, list of lines)
            """

            hist = P.hist(data, nbins, normed=1, align='center')
            data_range = [N.min(data), N.max(data)]

            # x's
            x = hist[1]
            dx = x[expand_tails] - x[0] # how much to expand tails by
            x = N.hstack((x[:expand_tails] - dx, x, x[-expand_tails:] + dx))

            nonparam = Nonparametric(data)
            # plot cdf
            if plot_cdf:
                P.plot(x, nonparam.cdf(x), 'k--', linewidth=1)

            p_thr = p

            data_p = _pvalue(data, nonparam.cdf, tail)
            data_p_thr = (data_p <= p_thr).ravel()

            x_p = _pvalue(x, Nonparametric(data).cdf, tail)
            x_p_thr = N.abs(x_p) <= p_thr
            # color bars which pass thresholding in red
            for thr, bar in zip(x_p_thr[expand_tails:], hist[2]):
                bar.set_facecolor(('w','r')[int(thr)])

            if not len(matches):
                # no matches were provided
                warning("No matching distributions were provided -- nothing to plot")
                return (hist, )

            lines = []
            labels = []
            for i in xrange(min(nbest, len(matches))):
                D, dist_gen, dist_name, params = matches[i]
                dist = getattr(scipy.stats, dist_gen)(*params)

                label = '%s' % (dist_name)
                if legend > 1: label += '(D=%.2f)' % (D)

                xcdf_p = N.abs(_pvalue(x, dist.cdf, tail))
                xcdf_p_thr = (xcdf_p <= p_thr).ravel()

                if p is not None and legend > 2:
                    # We need to compare detection under given p
                    data_cdf_p = N.abs(_pvalue(data, dist.cdf, tail))
                    data_cdf_p_thr = (data_cdf_p <= p_thr).ravel()

                    # true positives
                    tp = N.logical_and(data_cdf_p_thr, data_p_thr)
                    # false positives
                    fp = N.logical_and(data_cdf_p_thr, ~data_p_thr)
                    # false negatives
                    fn = N.logical_and(~data_cdf_p_thr, data_p_thr)

                    label += ' tp/fp/fn=%d/%d/%d)' % \
                            tuple(map(N.sum, [tp,fp,fn]))

                pdf = dist.pdf(x)
                line = P.plot(x, pdf, '-', linewidth=2, label=label)
                color = line[0].get_color()

                if plot_cdf:
                    cdf = dist.cdf(x)
                    P.plot(x, cdf, ':', linewidth=1, color=color, label=label)

                # TODO: decide on tp/fp/fn by not centers of the bins but
                #       by the values in data in the ranges covered by
                #       those bins. Then it would correspond to the values
                #       mentioned in the legend
                if p is not None:
                    # true positives
                    xtp = N.logical_and(xcdf_p_thr, x_p_thr)
                    # false positives
                    xfp = N.logical_and(xcdf_p_thr, ~x_p_thr)
                    # false negatives
                    xfn = N.logical_and(~xcdf_p_thr, x_p_thr)

                    # no need to plot tp explicitely -- marked by color of the bar
                    # P.plot(x[xtp], pdf[xtp], 'o', color=color)
                    P.plot(x[xfp], pdf[xfp], '^', color=color)
                    P.plot(x[xfn], pdf[xfn], 'v', color=color)

                lines.append(line)
                labels.append(label)

            if legend:
                P.legend(lines, labels)

            return (hist, lines)

    #if True:
    #    data = N.random.normal(size=(1000,1));
    #    matches = matchDistribution(
    #        data,
    #        distributions=['scipy',
    #                       ('norm', {'name':'norm_known',
    #                                 'scale': 1.0,
    #                                 'loc': 0.0})],
    #        nsamples=30, test='p-roc', p=0.05)
    #    P.figure(); plotDistributionMatches(data, matches, nbins=101,
    #                                        p=0.05, legend=4, nbest=5)


def autoNullDist(dist):
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
        a = N.ravel(a)
        outaxis = 0
    else:
        a = N.asarray(a)
        outaxis = axis
    return a, outaxis

def nanmean(x, axis=0):
    """Compute the mean over the given axis ignoring nans.

    :Parameters:
        x : ndarray
            input array
        axis : int
            axis along which the mean is computed.

    :Results:
        m : float
            the mean."""
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]
    factor = 1.0-N.sum(N.isnan(x),axis)*1.0/Norig

    x[N.isnan(x)] = 0
    return N.mean(x,axis)/factor
