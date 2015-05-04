# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Loaders for PyMVPA's own demo datasets"""

__docformat__ = 'restructuredtext'

import os
import numpy as np
from mvpa2.base import externals
from mvpa2 import pymvpa_dataroot, pymvpa_datadbroot


def load_example_fmri_dataset(name='1slice', literal=False):
    """Load minimal fMRI dataset that is shipped with PyMVPA."""
    from mvpa2.datasets.sources.openfmri import OpenFMRIDataset
    from mvpa2.datasets.mri import fmri_dataset
    from mvpa2.misc.io import SampleAttributes

    basedir = os.path.join(pymvpa_dataroot, 'openfmri')
    mask = {'1slice': os.path.join(pymvpa_dataroot, 'mask.nii.gz'),
            '25mm': os.path.join(basedir, 'sub001', 'masks', '25mm',
                                 'brain.nii.gz')}[name]

    if literal:
        model = 1
        subj = 1
        openfmri = OpenFMRIDataset(basedir)
        ds = openfmri.get_model_bold_dataset(model, subj, flavor=name,
                                             mask=mask, noinfolabel='rest')
        # re-imagine the global time_coords of a concatenated time series
        # this is only for the purpose of keeping the example data in the
        # exact same shape as it has always been. in absolute terms this makes no
        # sense as there is no continuous time in this dataset
        ds.sa['run_time_coords'] = ds.sa.time_coords
        ds.sa['time_coords'] = np.arange(len(ds)) * 2.5
    else:
        if name == '25mm':
            raise ValueError("The 25mm dataset is no longer available with "
                             "numerical labels")
        attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes.txt'))
        ds = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                          targets=attr.targets, chunks=attr.chunks,
                          mask=mask)

    return ds


def load_datadb_tutorial_data(path=os.path.join(
    pymvpa_datadbroot, 'tutorial_data', 'tutorial_data', 'data'),
        roi='brain', add_fa=None):
    """Loads the block-design demo dataset from PyMVPA dataset DB.

    Parameters
    ----------
    path : str
      Path of the directory containing the dataset files.
    roi : str or int or tuple or None
      Region Of Interest to be used for masking the dataset. If a string is
      given a corresponding mask image from the demo dataset will be used
      (mask_<str>.nii.gz). If an int value is given, the corresponding ROI
      is determined from the atlas image (mask_hoc.nii.gz). If a tuple is
      provided it may contain int values that a processed as explained
      before, but the union of a ROIs is taken to produce the final mask.
      If None, no masking is performed.
    add_fa : dict
      Passed on to the dataset creator function (see fmri_dataset() for
      more information).
    """
    import nibabel as nb
    from mvpa2.datasets.sources.openfmri import OpenFMRIDataset
    model = subj = 1
    dhandle = OpenFMRIDataset(path)
    maskpath = os.path.join(path, 'sub001', 'masks', 'orig')
    if roi is None:
        mask = None
    elif isinstance(roi, str):
        mask = os.path.join(maskpath, roi + '.nii.gz')
    elif isinstance(roi, int):
        nimg = nb.load(os.path.join(maskpath, 'hoc.nii.gz'))
        tmpmask = nimg.get_data() == roi
        mask = nb.Nifti1Image(tmpmask.astype(int), nimg.get_affine(),
                              nimg.get_header())
    elif isinstance(roi, tuple) or isinstance(roi, list):
        nimg = nb.load(os.path.join(maskpath, 'hoc.nii.gz'))
        if externals.versions['nibabel'] >= '1.2':
            img_shape = nimg.shape
        else:
            img_shape = nimg.get_shape()
        tmpmask = np.zeros(img_shape, dtype='bool')
        for r in roi:
            tmpmask = np.logical_or(tmpmask, nimg.get_data() == r)
        mask = nb.Nifti1Image(tmpmask.astype(int), nimg.get_affine(),
                              nimg.get_header())
    elif isinstance(roi, nb.Nifti1Image):
        mask = roi
    else:
        raise ValueError("Got something as mask that I cannot handle.")
    ds = dhandle.get_model_bold_dataset(model, subj, mask=mask, add_fa=add_fa,
                                        noinfolabel='rest')
    # fixup time_coords to make the impression of a continuous time series
    # this is only necessary until we have changed the tutorial to
    # show/encourage run-wise processing
    ds.sa['time_coords'] = np.linspace(0, (len(ds) * 2.5), len(ds) + 1)[:-1]
    return ds
