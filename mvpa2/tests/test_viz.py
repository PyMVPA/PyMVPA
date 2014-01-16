# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for visualizations'''


from mvpa2.testing import *
from nose.tools import *

skip_if_no_external('pylab')

import numpy as np

def test_hist():
    from mvpa2.viz import hist
    from mvpa2.misc.data_generators import normal_feature_dataset
    from matplotlib.axes import Subplot
    ds = normal_feature_dataset(10, 3, 10, 5)
    plots = hist(ds, ygroup_attr='targets', xgroup_attr='chunks',
                 noticks=None, xlim=(-.5, .5), normed=True)
    assert_equal(len(plots), 15)
    for sp in plots:
        assert_is_instance(sp, Subplot)
    # simple case
    plots = hist(ds)
    assert_equal(len(plots), 1)
    assert_is_instance(plots[0], Subplot)
    # make sure it works with plan arrays too
    plots = hist(ds.samples)
    assert_equal(len(plots), 1)
    assert_is_instance(plots[0], Subplot)

def test_imshow():
    from mvpa2.viz import matshow
    from mvpa2.misc.data_generators import normal_feature_dataset
    from matplotlib.colorbar import Colorbar
    ds = normal_feature_dataset(10, 2, 18, 5)
    im = matshow(ds)
    assert_is_instance(im.colorbar[0], Colorbar)
