# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Fixer for rdist in scipy
"""
# For scipy import
from __future__ import absolute_import

__docformat__ = 'restructuredtext'

from mvpa2.base import externals, warning, cfg

if __debug__:
    from mvpa2.base import debug

if externals.exists('scipy', raise_=True):
    import scipy
    import scipy.stats
    import scipy.stats as stats

if not externals.exists('good scipy.stats.rdist'):
    if __debug__:
        debug("EXT", "Fixing up scipy.stats.rdist")
    # Lets fix it up, future imports of scipy.stats should carry fixed
    # version, isn't python is \emph{evil} ;-)
    import numpy as np

    from scipy.stats.distributions import rv_continuous
    from scipy import special
    import scipy.integrate

    # NB: Following function is copied from scipy SVN rev.5236
    #     and fixed with pow -> np.power (thanks Josef!)
    # FIXME: PPF does not work.
    class rdist_gen(rv_continuous):
        def _pdf(self, x, c):
            return np.power((1.0-x*x),c/2.0-1) / special.beta(0.5,c/2.0)
        def _cdf_skip(self, x, c):
            #error inspecial.hyp2f1 for some values see tickets 758, 759
            return 0.5 + x/special.beta(0.5,c/2.0)* \
                   special.hyp2f1(0.5,1.0-c/2.0,1.5,x*x)
        def _munp(self, n, c):
            return (1-(n % 2))*special.beta((n+1.0)/2,c/2.0)

    # Lets try to avoid at least some of the numerical problems by removing points
    # around edges
    rdist = rdist_gen(a=-1.0, b=1.0, name="rdist", longname="An R-distributed",
                      shapes="c", extradoc="""

    R-distribution

    rdist.pdf(x,c) = (1-x**2)**(c/2-1) / B(1/2, c/2)
    for -1 <= x <= 1, c > 0.
    """
                      )
    # Fix up number of arguments for veccdf's vectorize
    if rdist.veccdf.nin == 1:
        if __debug__:
            debug("EXT", "Fixing up veccdf.nin to make 2 for rdist")
        rdist.veccdf.nin = 2

    scipy.stats.distributions.rdist_gen = scipy.stats.rdist_gen = rdist_gen
    scipy.stats.distributions.rdist = scipy.stats.rdist = rdist

    try: # Retest
        externals.exists('good scipy.stats.rdist', force=True,
                         raise_=True)
    except RuntimeError:
        warning("scipy.stats.rdist was not fixed with a monkey-patch. "
                "It remains broken")
    # Revert so if configuration stored, we know the true flow of things ;)
    cfg.set('externals', 'have good scipy.stats.rdist', 'no')


if not externals.exists('good scipy.stats.rv_discrete.ppf'):
    # Local rebindings for ppf7 (7 is for the scipy version from
    # which code was borrowed)
    arr = np.asarray
    from scipy.stats.distributions import valarray, argsreduce
    from numpy import shape, place, any

    def ppf7(self,q,*args,**kwds):
        """
        Percent point function (inverse of cdf) at q of the given RV

        Parameters
        ----------
        q : array-like
            lower tail probability
        arg1, arg2, arg3,... : array-like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array-like, optional
            location parameter (default=0)

        Returns
        -------
        k : array-like
            quantile corresponding to the lower tail probability, q.

        """
        loc = kwds.get('loc')
        args, loc = self._rv_discrete__fix_loc(args, loc)
        q,loc  = map(arr,(q,loc))
        args = tuple(map(arr,args))
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q==1) & cond0
        cond = cond0 & cond1
        output = valarray(shape(cond),value=self.badvalue,typecode='d')
        #output type 'd' to handle nin and inf
        place(output,(q==0)*(cond==cond), self.a-1)
        place(output,cond2,self.b)
        if any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output,cond,self._ppf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    scipy.stats.distributions.rv_discrete.ppf = ppf7
    try:
        externals.exists('good scipy.stats.rv_discrete.ppf', force=True,
                         raise_=True)
    except RuntimeError:
        warning("rv_discrete.ppf was not fixed with a monkey-patch. "
                "It remains broken")
    cfg.set('externals', 'have good scipy.stats.rv_discrete.ppf', 'no')

if externals.versions['scipy'] >= '0.8.0' and \
       not externals.exists('good scipy.stats.rv_continuous._reduce_func(floc,fscale)'):
    if __debug__:
        debug("EXT", "Fixing up scipy.stats.rv_continuous._reduce_func")

    # Borrowed from scipy v0.4.3-5978-gce90df2
    # Copyright: 2001, 2002 Enthought, Inc.; 2003-2012 SciPy developers
    # License: BSD-3
    def _reduce_func_fixed(self, args, kwds):
        args = list(args)
        Nargs = len(args)
        fixedn = []
        index = range(Nargs)
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in zip(index, names):
            if kwds.has_key(key):
                fixedn.append(n)
                args[n] = kwds[key]
            else:
                x0.append(args[n])

        if len(fixedn) == 0:
            func = self.nnlf
            restore = None
        else:
            if len(fixedn) == len(index):
                raise ValueError("All parameters fixed. There is nothing to optimize.")
            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return self.nnlf(newtheta, x)

        return x0, func, restore, args

    stats.rv_continuous._reduce_func = _reduce_func_fixed
