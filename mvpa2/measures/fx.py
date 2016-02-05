# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Estimate featurewise measures by applying a function along samples."""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug

import inspect, re
import numpy as np

from mvpa2.base import externals
from mvpa2.base.dochelpers import _repr_attrs

from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.misc.attrmap import AttributeMap
from mvpa2.datasets import Dataset

class BinaryFxFeaturewiseMeasure(FeaturewiseMeasure):
    """A feature-wise measure to be computed against a samples attribute
    """

    is_trained = True
    def __init__(self, fx, space='targets', uni=True, numeric=False, **kwargs):
        """
        Parameters
        ----------
        numeric: bool, optional
          Either attribute (e.g. 'targets') values should be converted to
          numeric values before computation
        """
        super(BinaryFxFeaturewiseMeasure, self).__init__(space=space, **kwargs)
        self.fx = fx
        self.uni = uni
        self.numeric = numeric

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(BinaryFxFeaturewiseMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['fx'])
            + _repr_attrs(self, ['space'], default='targets')
            + _repr_attrs(self, ['uni'], default=True)
            + _repr_attrs(self, ['numeric'], default=False)
            )

    def __str__(self, *args, **kwargs):
        fx_str = str(self.fx)
        # TODO: unify within dochelpers functionality?
        if fx_str.startswith("<function") and hasattr(self.fx, '__name__'):
            fx_str = self.fx.__name__
        if fx_str == "<lambda>":
            try:
                # where feasible -- extract actual code
                fx_str = inspect.getsource(self.fx).strip()
                # tidy it up
                fx_str = re.sub("^\s*[A-Za-z_]*\s*=\s*", "", fx_str)
            except:
                pass
        return super(BinaryFxFeaturewiseMeasure, self).__str__(fx_str, *args, **kwargs)

    def _call(self, ds):
        y = ds.sa[self.space].value
        if self.numeric or ((self.numeric is None) and y.dtype.char == 'S'):
            y = AttributeMap().to_numeric(y)
        # TODO:
        if not self.uni:
            out = self.fx(ds.samples, y)
        else:
            out = np.array([self.fx(feat, y) for feat in ds.samples.T])
        return Dataset(out[None], fa=ds.fa)

if externals.exists('statsmodels'):
    import statsmodels.sandbox.distributions.mv_measures as mv_measures

    # TODO move under .support somewhere and submit a PR with fix to statsmodels
    def mutualinfo_kde(x, y):
        """A lean adapter for statsmodels mutualinfo_kde which retuns 0 for nan results
        """
        res = mv_measures.mutualinfo_kde(x, y)
        if np.isnan(res):
            return 0  # could get out of bounds
        return res

    def targets_mutualinfo_kde(attr='targets'):
        """Compute mutual information between each features and a samples attribute (targets)

        Uses kernel density estimation for MI calculations
        """
        return BinaryFxFeaturewiseMeasure(mutualinfo_kde, space=attr,
                                          uni=True, numeric=True)

    """
    TODO:
    Doesn't work with "toy" test data

    $> nosetests -s -v --pdb --pdb-failures mvpa2/measures/fx.py:test_BinaryFxFeatureMeasure_mi
mvpa2.measures.fx.test_BinaryFxFeatureMeasure_mi ... dcor: [[ 0.99998731]]
mi b: > /usr/lib/python2.7/dist-packages/statsmodels/sandbox/distributions/mv_measures.py(106)mutualinfo_binned()
-> shift[0] -= 2*1e-6
(Pdb) print y
[ 0.    0.99  0.    1.  ]
(Pdb) print x
[ 0.  1.  0.  1.]

# implementation seems to be too raw to be useful
    def targets_mutualinfo_binned(attr='targets', bins='auto'):
        \"""Compute mutual information between each features and a samples attribute (targets)

        Bins input for MI calculations
        \"""
        return BinaryFxFeaturewiseMeasure(
            lambda x,y: mv_measures.mutualinfo_binned(x, y, bins), space=attr,
            uni=True, numeric=True)
"""

from mvpa2.misc.dcov import dcorcoef
def targets_dcorrcoef(attr='targets'):
    """Return dCorr coefficient between each feature and a samples attribute (targets)
    """
    return BinaryFxFeaturewiseMeasure(dcorcoef, space=attr, uni=True, numeric=True)

# TODO: RF CorrCoef to use BinaryFxMeasure

