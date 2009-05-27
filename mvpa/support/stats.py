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

from mvpa.base import externals

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

