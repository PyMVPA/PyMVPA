# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for basic mappers'''

import numpy as np

from mvpa2.testing.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true, assert_array_equal, assert_array_less
from mvpa2.measures.corrstability import CorrStability
from mvpa2.testing.datasets import datasets
from mvpa2.testing import sweepargs

@sweepargs(ds=datasets.itervalues())
def test_corrstability_smoketest(ds):
    if not 'chunks' in ds.sa:
        return
    if len(ds.sa['targets'].unique) > 30:
        # was regression dataset
        return
    # very basic testing since
    cs = CorrStability()
    #ds = datasets['uni2small']
    out = cs(ds)
    assert_equal(out.shape, (ds.nfeatures,))
    ok_(np.all(out >= -1.001))  # it should be a correlation after all
    ok_(np.all(out <= 1.001))

    # and theoretically those nonbogus features should have higher values
    if 'nonbogus_targets' in ds.fa:
        bogus_features = np.array([x==None for x in  ds.fa.nonbogus_targets])
        assert_array_less(np.mean(out[bogus_features]), np.mean(out[~bogus_features]))
    # and if we move targets to alternative location
    ds = ds.copy(deep=True)
    ds.sa['alt'] = ds.T
    ds.sa.pop('targets')
    assert_raises(KeyError, cs, ds)
    cs = CorrStability('alt')
    out_ = cs(ds)
    assert_array_equal(out, out_)


