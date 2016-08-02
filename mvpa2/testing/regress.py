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

from os.path import join as pathjoin

import hashlib
import mvpa2

from mvpa2 import pymvpa_dataroot, externals

def get_testing_fmri_dataset_filename():
    """Generate path to the testing filename based on mvpa2/nibabel versions
    """
    # explicitly so we do not anyhow depend on dict ordering
    versions_hash = hashlib.md5(
        "_".join(["%s:%s" % (k, externals.versions[k])
                  for k in sorted(externals.versions)])
    ).hexdigest()[:6]

    filename = 'mvpa-%s_nibabel-%s-%s.hdf5' % (
        mvpa2.__version__,
        externals.versions['nibabel'],
        versions_hash)

    return pathjoin(pymvpa_dataroot, 'testing', 'fmri_dataset', filename)

get_testing_fmri_dataset_filename.__test__ = False


def generate_testing_fmri_dataset(filename=None):
    """Helper to generate a dataset for regression testing of mvpa2/nibabel

    Parameters
    ----------
    filename : str
       Filename of a dataset file to store.  If not provided, it is composed
       using :func:`get_testing_fmri_dataset_filename`

    Returns
    -------
    Dataset, string
       Generated dataset, filename to the HDF5 where it was stored
    """
    import mvpa2
    from mvpa2.base.hdf5 import h5save
    from mvpa2.datasets.sources import load_example_fmri_dataset
    # Load our sample dataset
    ds_full = load_example_fmri_dataset(name='1slice', literal=False)
    # Subselect a small "ROI"
    ds = ds_full[20:23, 10:14]
    # collect all versions/dependencies for possible need to troubleshoot later
    ds.a['wtf'] = mvpa2.wtf()
    ds.a['versions'] = mvpa2.externals.versions
    # save to a file identified by version of PyMVPA and nibabel and hash of
    # all other versions
    out_filename = filename or get_testing_fmri_dataset_filename()
    h5save(out_filename, ds, compression=9)
    # ATM it produces >700kB .hdf5 which is this large because of
    # the ds.a.mapper with both Flatten and StaticFeatureSelection occupying
    # more than 190kB each, with ds.a.mapper as a whole generating 570kB file
    # Among those .ca seems to occupy notable size, e.g. 130KB for the FlattenMapper
    # even though no heavy storage is really needed for any available value --
    # primarily all is meta-information embedded into hdf5 to describe our things
    return ds, out_filename

generate_testing_fmri_dataset.__test__ = False

if __name__ == '__main__':
    generate_testing_fmri_dataset()
