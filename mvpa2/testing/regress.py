#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Convinience functions to generate/update datasets for regression testing
"""

__docformat__ = 'restructuredtext'

import os

import mvpa2

from mvpa2 import pymvpa_dataroot, externals

def get_testing_fmri_dataset_filename():
    """Generate path to the testing filename based on mvpa2/nibabel versions
    """
    filename = 'mvpa-%s_nibabel-%s.hdf5' % (mvpa2.__version__, externals.versions['nibabel'])
    return os.path.join(pymvpa_dataroot, 'testing', 'fmri_dataset', filename)

def generate_testing_fmri_dataset():
    """Helper to generate a dataset for regression testing of mvpa2/nibabel

    Returns
    -------
    Dataset, string
       Generated dataset, filename to the HDF5 where it was stored
    """
    import mvpa2
    from mvpa2.base.hdf5 import h5save
    from mvpa2.misc.data_generators import load_example_fmri_dataset
    # Load our sample dataset
    ds_full = load_example_fmri_dataset(name='1slice', literal=False)
    # Subselect a small "ROI"
    ds = ds_full[20:23, 10:14]
    # collect all versions/dependencies for possible need to troubleshoot later
    # but only via WTF string, due to https://github.com/PyMVPA/PyMVPA/issues/266
    ds.a['wtf'] = mvpa2.wtf()
    # save to a file identified by version of PyMVPA and nibabel
    filename = get_testing_fmri_dataset_filename()
    h5save(filename, ds, compression=9)
    # ATM it produces 680kB .hdf5 which is this large because of
    # the ds.a.mapper with both Flatten and StaticFeatureSelection occupying
    # more than 190kB each, with ds.a.mapper as a whole generating 570kB file
    # Among those .ca seems to occupy notable size, e.g. 130KB for the FlattenMapper
    # even though no heavy storage is really needed for any available value --
    # primarily all is meta-information embedded into hdf5 to describe our things
    return ds, filename

if __name__ == '__main__':
    generate_testing_fmri_dataset()