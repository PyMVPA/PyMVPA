# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for visualizations'''


from mvpa2 import pymvpa_dataroot
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
    # old mpl returns a tuple of Colorbar which is anyways available as its .ax
    if isinstance(im.colorbar, tuple):
        assert_is_instance(im.colorbar[0], Colorbar)
        assert_true(im.colorbar[1] is im.colorbar[0].ax)
    else:
        # new mpls do it withough unnecessary duplication
        assert_is_instance(im.colorbar, Colorbar)

def test_lightbox():
    skip_if_no_external('nibabel') # used for loading the niftis here
    # smoketest for lightbox - moved from its .py __main__
    from mvpa2.misc.plot.lightbox import plot_lightbox
    fig = plot_lightbox(
        #background = NiftiImage('%s/anat.nii.gz' % impath),
        background = os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
        background_mask = None,
        overlay = os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
        overlay_mask = os.path.join(pymvpa_dataroot, 'mask.nii.gz'),
        #
        do_stretch_colors = False,
        add_colorbar = True,
        cmap_bg = 'gray',
        cmap_overlay = 'hot', # YlOrRd_r # pl.cm.autumn
        #
        fig = None,
        # vlim describes value limits
        # clim color limits (same by default)
        vlim = [1500, None],
        #vlim_type = 'symneg_z',
        interactive = True,
        #
        #nrows = 2,
        #ncolumns = 3,
        add_info = (1, 2),
        add_hist = (0, 2),
        #
        slices = [0]
        )
    assert_true(fig)