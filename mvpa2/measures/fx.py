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

    def __repr__(self, prefixes=[]):
        return super(BinaryFxFeaturewiseMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['fx'])
            + _repr_attrs(self, ['space'], default='targets')
            + _repr_attrs(self, ['uni'], default=True)
            + _repr_attrs(self, ['numeric'], default=False)
            )

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
        res = mv_measures.mutualinfo_kde(x, y)
        if np.isnan(res):
            return 0 # could get out of bounds
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

import numpy as np

from mvpa2.datasets import dataset_wizard
from mvpa2.testing import sweepargs
from mvpa2.testing.datasets import datasets as tdatasets
from mvpa2.testing import assert_array_equal, assert_array_less, assert_equal

@sweepargs(ds=tdatasets.itervalues())
def test_BinaryFxFeatureMeasure(ds):
    if not isinstance(ds.samples, np.ndarray):
        return
    # some simple function
    f = lambda x, y: np.sum((x.T*y).T, axis=0)
    fx = BinaryFxFeaturewiseMeasure(f, uni=False, numeric=True)
    fx_uni = BinaryFxFeaturewiseMeasure(f, uni=True, numeric=True)
    out = fx(ds)
    out_uni = fx_uni(ds)
    assert(len(out) == 1)
    assert_array_equal(out.samples, out_uni)
    assert_equal(out.fa, out_uni.fa)

_nonlin_tests = [(dataset_wizard([0, 1-0.01, 0, 1],
                                      targets=['a', 'b', 'a', 'b']),
                                     ([0.99], [1])),
                      (dataset_wizard([0, 1-0.01, 2, 0, 1, 2],
                                      targets=['a', 'b', 'c', 'a', 'b', 'c']),
                                     ([0.99], [1])),
                      # verify that order of 'labels' doesn't matter to get the same correspondence
                      (dataset_wizard([1-0.01, 0, 1, 0],
                                      targets=['a', 'b', 'a', 'b']),
                                     ([0.99], [1])),
                      # unfortunately with both normal kde based MI and dcorr
                      # we are not getting "ideal" results in case of "non-linear"
                      # but strict dependencies
                      (dataset_wizard([0, 1-0.01, 2, 0, 1, 2],
                                      targets=['a', 'c', 'b', 'a', 'c', 'b']),
                                     ([0.8], [1])),
                      # 2nd feature should have no information above the targets
                      (dataset_wizard([[0, 1-0.01],[1-0.01, 0], [0, 0], [1, 1]],
                                      targets=['a', 'b', 'a', 'b']),
                                     ([0.99, -1e-10], [1.+1e-10, 0.01])),
                        ]

_nonlin_tests_fx = [targets_dcorrcoef()]
if externals.exists('statsmodels'):
    # -0.01 because current implementation would run into a singular matrix if y is exactly
    # the x. TODO: report/fix
    _nonlin_tests_fx.append(targets_mutualinfo_kde())


@sweepargs(ds_mi=_nonlin_tests)
@sweepargs(fx=_nonlin_tests_fx)
def test_BinaryFxFeatureMeasure_mi(fx, ds_mi):
    ds, (mi_min, mi_max) = ds_mi
    #fx = BinaryFxFeaturewiseMeasure(mutualinfo_kde, numeric=True)
    #print 'dcor:', targets_dcorrcoef()(ds).samples
    #print 'mi b:', targets_mutualinfo_binned()(ds).samples
    res = fx(ds)
    assert_equal(res.shape, (1, ds.nfeatures))
    mi = res.samples[0]
    assert_array_less(mi_min, mi)
    assert_array_less(mi, mi_max)
