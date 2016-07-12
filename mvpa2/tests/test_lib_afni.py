# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA atlases"""

import numpy as np
from mvpa2.testing import *

skip_if_no_external('nibabel')
import nibabel as nb
from mvpa2.support.afni import lib_prep_afni_surf



@sweepargs(mock_3dinfo=[False, True])
@sweepargs(is_plumb=[False, True])
@with_tempfile('.nii', 'test_plump')
def test__ensure_expvol_is_plump(filename, is_plumb, mock_3dinfo):
    if not mock_3dinfo:
        # AFNI's 3dinfo is required
        skip_if_no_external('afni-3dinfo')

    data = np.random.normal(size=(2, 2, 3, 3)).astype(np.int16)

    affine = np.eye(4)
    if not is_plumb:
        affine[0, 1] = 2.

    img = nb.Nifti1Image(data, affine)
    img.to_filename(filename)

    if mock_3dinfo:
        # mock AFNIs 3dinfo
        plump_str = 'Plumb' if is_plumb else "oblique"
        mocked_info_lines = 'Data Axes Tilt:  ' + plump_str
        mocked_3dinfo = lambda: mocked_info_lines
        func = lambda: lib_prep_afni_surf._ensure_expvol_is_plump(None,
                                                                  mocked_3dinfo)
    else:
        func = lambda: lib_prep_afni_surf._ensure_expvol_is_plump(filename)

    if is_plumb:

        # calling func should be ok
        func()
    else:
        # calling func should raise an Exception
        assert_raises(ValueError, func)
