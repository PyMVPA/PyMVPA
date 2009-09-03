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

__docformat__ = 'restructuredtext'

from mvpa.base import externals, warning

if __debug__:
    from mvpa.base import debug

if externals.exists('scipy', raiseException=True):
    import scipy
    import scipy.stats
    import scipy.stats as stats

if not externals.exists('good scipy.stats.rdist'):
    if __debug__:
        debug("EXT", "Fixing up scipy.stats.rdist")
    # Lets fix it up, future imports of scipy.stats should carry fixed
    # version, isn't python is \emph{evil} ;-)
    import numpy as N

    from scipy.stats.distributions import rv_continuous
    from scipy import special
    import scipy.integrate

    # NB: Following function is copied from scipy SVN rev.5236
    #     and fixed with pow -> N.power (thanks Josef!)
    # FIXME: PPF does not work.
    class rdist_gen(rv_continuous):
        def _pdf(self, x, c):
            return N.power((1.0-x*x),c/2.0-1) / special.beta(0.5,c/2.0)
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

    externals.exists('good scipy.stats.rdist', force=True)
    try:
        externals.exists('good scipy.stats.rdist', raiseException=True)
    except RuntimeError:
        warning("scipy.stats.rdist was not fixed with a monkey-patch. "
                "It remains broken")


if not externals.exists('good scipy.stats.rv_discrete.ppf'):
    # Local rebindings for ppf7 (7 is for the scipy version from
    # which code was borrowed)
    arr = N.asarray
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
    externals.exists('good scipy.stats.rv_discrete.ppf', force=True)
    try:
        externals.exists('good scipy.stats.rv_discrete.ppf', raiseException=True)
    except RuntimeError:
        warning("rv_discrete.ppf was not fixed with a monkey-patch. "
                "It remains broken")
