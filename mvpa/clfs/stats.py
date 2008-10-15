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


def _pvalue(x, cdf_func, tail):
    """Helper function to return p-value(x) given cdf and tail
    """
    is_scalar = N.isscalar(x)
    if is_scalar:
        x = [x]

    cdf = cdf_func(x)
    if tail == 'left':
        pass
    elif tail == 'right':
        cdf = 1 - cdf
    elif tail == 'any':
        right_tail = (cdf >= 0.5)
        cdf[right_tail] = 1.0 - cdf[right_tail]
        # we need to half the signficance
        cdf *= 2

    if is_scalar: return cdf[0]
    else:         return cdf


class NullDist(Stateful):
    """Base class for null-hypothesis testing.

    """

    # Although base class is not benefiting from states, derived
    # classes do (MCNullDist). For the sake of avoiding multiple
    # inheritance and associated headache -- let them all be Stateful,
    # performance hit should be negligible in most of the scenarios
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
        if tail not in ['left', 'right', 'any']:
            raise ValueError, 'Unknown value "%s" to `tail` argument.' \
                  % tail


    def fit(self, measure, wdata, vdata=None):
        """Implement to fit the distribution to the data."""
        raise NotImplementedError


    def cdf(self, x):
        """Implementations return the value of the cumulative distribution
        function (left or right tail dpending on the setting).
        """
        raise NotImplementedError


    def p(self, x):
        """Returns the p-value for values of `x`.
        Returned values are determined left, right, or from any tail
        depending on the constructor setting.

        In case a `FeaturewiseDatasetMeasure` was used to estimate the
        distribution the method returns an array. In that case `x` can be
        a scalar value or an array of a matching shape.
        """
        return _pvalue(x, self.cdf, self._tail)



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
        NullDist.__init__(self, **kwargs)

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


if externals.exists('scipy'):
    import scipy.stats
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
            fargs = (loc, scale)
            if args is not None:
                fargs += args
            self._fargs = fargs
            """Arguments which should get some fixed value"""


        def nnlf(self, theta, x):
            # - sum (log pdf(x, theta),axis=0)
            #   where theta are the parameters (including loc and scale)
            #
            try:
                fargs = self._fargs
                i=-1
                if fargs[1] is not None:
                    scale = fargs[1]
                else:
                    scale = theta[i]
                    i -= 1

                if fargs[0] is not None: loc = fargs[0]
                else:
                    loc = theta[i]
                    i -= 1
                # TODO: if we want to fix params as well -- do here
                args = tuple(theta[:i+1])
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
            fargs = self._fargs
            Narg = len(args)
            if Narg != self.numargs:
                if Narg > self.numargs:
                    raise ValueError, "Too many input arguments."
                else:
                    args += (1.0,)*(self.numargs-Narg)
            # TODO: if needed to fix arguments -- remove them here
            # location and scale are at the end
            x0 = args
            if fargs[0] is None: x0 = x0 + (loc0,)
            if fargs[1] is None: x0 = x0 + (scale0,)
            opt_x = scipy.optimize.fmin(self.nnlf, x0, args=(N.ravel(data),), disp=0)
            if fargs[1] is not None: opt_x = N.hstack((opt_x,fargs[1:2]))
            if fargs[0] is not None:
                opt_x = N.hstack((opt_x[:-1], fargs[0:1],opt_x[-1:]))
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
                          test='kstest', distributions=None,
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
               parameters: `p=0.05` and `tail='any'`
             'kstest' : 'full-body' distribution comparison. The best
               choice is made by minimal reported distance after estimating
               parameters of the distribution. Parameter `p=0.05` sets
               threshold to reject null-hypothesis that distribution is the
               same.
               WARNING: older versions (e.g. 0.5.2 in etch) of scipy have
                        incorrect kstest implementation and do not function
                        properly
          distributions : None or list of basestring
            Distributions to check. If None, all known in scipy.stats are
            tested.
          **kwargs
            Additional arguments which are needed for each particular test
            (see above)
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
            tail = kwargs.get('tail', 'any')
            data_p = _pvalue(data, Nonparametric(data).cdf, tail)
            data_p_thr = data_p <= p_thr
            true_positives = N.sum(data_p_thr)
            if true_positives == 0:
                raise ValueError, "Provided data has no elements in non-" \
                      "parametric distribution under p<=%.3f. Please " \
                      "increase the size of data or value of p" % p
            if __debug__:
                debug('STAT_', 'Number of positives in non-parametric '
                      'distribution is %d' % true_positives)

        if distributions is None:
            distributions = scipy.stats.distributions.__all__
        results = []
        for d in distributions:
            # perform actions which might puke for some distributions
            try:
                dist_gen = getattr(scipy.stats, d)
                # specify distribution 'optimizer'. If loc or scale was provided,
                # use home-brewed rv_semifrozen
                if loc is not None or scale is not None:
                    dist_opt = rv_semifrozen(dist_gen, loc=loc, scale=scale)
                else:
                    dist_opt = dist_gen
                dist_params = dist_opt.fit(data_selected)
                if __debug__:
                    debug('STAT__',
                          'Got distribution parameters %s for %s'
                          % (dist_params, d))
                if test == 'p-roc':
                    cdf_func = lambda x: dist_gen.cdf(x, *dist_params)
                    # We need to compare detection under given p
                    cdf_p = _pvalue(data, cdf_func, tail)
                    cdf_p_thr = cdf_p <= p_thr
                    D, p = N.sum(N.abs(data_p_thr - cdf_p_thr))*1.0/true_positives, 1
                    if __debug__: res_sum = 'D=%.2f' % D
                elif test == 'kstest':
                    D, p = kstest(data, d, args=dist_params)
                    if __debug__: res_sum = 'D=%.3f p=%.3f' % (D, p)
            except Exception, e:
                if __debug__:
                    debug('STAT__',
                          'Testing for %s distribution failed due to %s'
                          % (d, str(e)))
                continue

            if p > p_thr and not N.isnan(D):
                results += [ (D, d, dist_params) ]
                if __debug__:
                    debug('STAT_', 'Testing for %s dist.: %s' % (d, res_sum))
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
            sresult = lambda r:'%s(%s)=%.2f' % (r[1], ', '.join(map(str, r[2])), r[0])
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
                                    p=None, tail='any'):
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
              tail : ('left', 'right', 'any')
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
            x_p_thr = x_p <= p_thr
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
                D, dist_name, params = matches[i]
                dist = getattr(scipy.stats, dist_name)(*params)

                label = '%s' % (dist_name)
                if legend > 1: label += '(D=%.2f)' % (D)

                xcdf_p = _pvalue(x, dist.cdf, tail)
                xcdf_p_thr = (xcdf_p <= p_thr).ravel()

                if p is not None and legend > 2:
                    # We need to compare detection under given p
                    data_cdf_p = _pvalue(data, dist.cdf, tail)
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
    #    data = N.random.normal(size=(1000,1)); test='p-roc';
    #    matches = matchDistribution(data, nsamples=30, test=test, p=0.05)
    #    P.figure(); plotDistributionMatches(data, matches, nbins=101, p=0.05, legend=4, nbest=5)

