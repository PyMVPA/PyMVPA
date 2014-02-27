# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA's data sources"""

from mvpa2.testing import *

def test_sklearn_data_wrappers():
    skip_if_no_external('skl')
    import mvpa2.datasets.sources as mvpads
    if externals.versions['skl'] >= '0.9':
        from sklearn import datasets as skldata
    else:
        from scikits.learn import datasets as skldata
    import inspect
    found_fx = 0
    for fx in skldata.__dict__:
        if not (fx.startswith('make_') or fx.startswith('load_')) \
                or fx in ['load_filenames', 'load_files',
                          'load_sample_image', 'load_sample_images',
                          'load_svmlight_files', 'load_svmlight_file']:
            continue
        found_fx += 1
        # fx() signatures must be the same
        assert_equal(inspect.getargspec(getattr(skldata, fx)),
                     inspect.getargspec(getattr(mvpads, 'skl_%s' % fx[5:])))
        if fx in ('load_iris',):
            # add this one if sklearn issue #2865 is resolved
            # 'load_boston'):
            assert_array_equal(getattr(skldata, fx)()['data'],
                               getattr(mvpads, 'skl_%s' % fx[5:])().samples)
    # if we do not get a whole bunch, something changed
    assert_true(found_fx > 15)
