# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA's GIFTI dataset"""

from mvpa2.base import externals
from mvpa2.testing import SkipTest

if not externals.exists('nibabel'):
    raise SkipTest

from nibabel.gifti import giftiio as nb_giftiio
from mvpa2.datasets.base import Dataset
from mvpa2.datasets.gifti import gifti_dataset, map2gifti

import numpy as np

from mvpa2.testing.tools import assert_datasets_almost_equal, \
    assert_datasets_equal, assert_raises, with_tempfile
from mvpa2.testing import sweepargs



def _get_test_sample_node_data(format_=None):
    # returns test data in various formats
    if format_ == None:
        samples = np.asarray(
            [[2.032, -0.892, -0.826, 1.163],
             [0.584, 1.844, 1.166, -0.848],
             [-1.444, -0.262, -1.921, 3.085],
             [-0.518, 2.339, 0.441, 1.856],
             [1.191, -0.204, -0.209, 1.755],
             [-1.326, 2.724, 0.148, 0.502]])
        nodes = np.asarray([8, 7, 2, 3])

    elif format_ == 'ASCII':
        np_samples, np_nodes, _ = _get_test_sample_node_data(format_=None)
        nfeatures = np_samples.shape[1]
        samples = [(('%.3f ' * nfeatures) % tuple(sample.tolist())).strip()
                   for sample in np_samples]
        nodes = ('%d ' * nfeatures) % tuple(np_nodes.tolist())

    elif format_ == 'Base64Binary':
        samples = ["SgwCQB1aZL+8dFO/L92UPw==",
                   "BoEVPzEI7D99P5U/hxZZvw==",
                   "/tS4v90khr5U4/W/pHBFQA==",
                   "ppsEvy2yFUDByuE+aJHtPw==",
                   "sHKYP2DlUL4ZBFa+16PgPw==",
                   "XrqpvwRWLkBQjRc+EoMAPw=="]
        nodes = "CAAAAAcAAAACAAAAAwAAAA=="

    elif format_ == 'GZipBase64Binary':
        samples = ["eJzz4mFykI1K2b+nJHi//t0p9gAsDAZU",
                   "eJxjaxS1N+R4Y19rP9W+XSxyPwAolAWF",
                   "eJz7d2XH/rsqbftCHn/dv6TA1QEAXukKEw==",
                   "eJxbNptlv+4mUYeDpx7aZUx8aw8AQboICA==",
                   "eJzbUDTDPuFpwD5JlrB91xc/sAcAQ0gIFw==",
                   "eJyL27VyP0uYnkNAr7idUDODPQA14AVP"]
        nodes = "eJzjYGBgYAdiJiBmBmIAAQAAFQ=="

    nfeatures = 4

    return samples, nodes, nfeatures



def _get_test_dataset(include_nodes=True):
    # returns test dataset matching the contents of _get_test_sample_node_data
    samples, nodes, _ = _get_test_sample_node_data()
    ds = Dataset(np.asarray(samples))

    if include_nodes:
        ds.fa['node_indices'] = np.asarray(nodes)

    nsamples = ds.nsamples
    ds.sa['intents'] = ['NIFTI_INTENT_NONE'] * nsamples

    return ds



def _build_gifti_string(format_, include_nodes=True):
    # builds the string contents of a GIFTI file
    samples, nodes, nfeatures = _get_test_sample_node_data(format_)
    nsamples = len(samples)

    ndata_arrays = nsamples + (1 if include_nodes else 0)

    header = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">
<GIFTI Version="1.0"  NumberOfDataArrays="%d">""" % (ndata_arrays)

    def _build_data_array(data, intent, encoding=format_, dim=nfeatures):
        datatype = "INT32" if intent == "NODE_INDEX" else "FLOAT32"

        return """<MetaData/>
                   <LabelTable/>
                   <DataArray  ArrayIndexingOrder="RowMajorOrder"
                               DataType="NIFTI_TYPE_%s"
                               Dim0="%d"
                               Dimensionality="1"
                               Encoding="%s"
                               Endian="LittleEndian"
                               ExternalFileName=""
                               ExternalFileOffset=""
                               Intent="NIFTI_INTENT_%s">
                      <MetaData>
                      </MetaData>
                      <Data>%s</Data>
                    </DataArray>
                """ % (datatype, dim, encoding, intent, data)

    body_elements = []
    if include_nodes:
        body_elements.append(_build_data_array(nodes, 'NODE_INDEX'))

    body_elements.extend(_build_data_array(sample, 'NONE')
                         for sample in samples)

    footer = """</GIFTI>"""

    return header + "".join(body_elements) + footer



@sweepargs(include_nodes=(False, True))
@sweepargs(format_=("ASCII", "Base64Binary", "GZipBase64Binary"))
@with_tempfile(suffix='.func.gii')
def test_gifti_dataset(fn, format_, include_nodes):
    expected_ds = _get_test_dataset(include_nodes)

    expected_ds_sa = expected_ds.copy(deep=True)
    expected_ds_sa.sa['chunks'] = [4, 3, 2, 1, 3, 2]
    expected_ds_sa.sa['targets'] = ['t%d' % i for i in xrange(6)]


    # build GIFTI file from scratch
    gifti_string = _build_gifti_string(format_, include_nodes)
    with open(fn, 'w') as f:
        f.write(gifti_string)

    # reading GIFTI file
    ds = gifti_dataset(fn)
    assert_datasets_almost_equal(ds, expected_ds)

    # test GiftiImage input
    img = nb_giftiio.read(fn)
    ds2 = gifti_dataset(img)
    assert_datasets_almost_equal(ds2, expected_ds)

    # test using Nibabel's output from write
    nb_giftiio.write(img, fn)
    ds3 = gifti_dataset(fn)
    assert_datasets_almost_equal(ds3, expected_ds)

    # test targets and chunks arguments
    ds3_sa = gifti_dataset(fn, targets=expected_ds_sa.targets,
                           chunks=expected_ds_sa.chunks)
    assert_datasets_almost_equal(ds3_sa, expected_ds_sa)

    # test map2gifti
    img2 = map2gifti(ds)
    ds4 = gifti_dataset(img2)
    assert_datasets_almost_equal(ds4, expected_ds)

    map2gifti(ds, fn, encoding=format_)
    ds5 = gifti_dataset(fn)
    assert_datasets_almost_equal(ds5, expected_ds)

    # test map2gifti with array input; nodes are not stored
    map2gifti(ds.samples, fn)
    ds6 = gifti_dataset(fn)
    if include_nodes:
        assert_raises(AssertionError, assert_datasets_almost_equal,
                      ds6, expected_ds)
    else:
        assert_datasets_almost_equal(ds6, expected_ds)

    assert_raises(TypeError, gifti_dataset, ds3_sa)
    assert_raises(TypeError, map2gifti, img, fn)



@sweepargs(include_nodes=(False, True))
@with_tempfile(suffix='.hdf5')
def test_gifti_dataset_h5py(fn, include_nodes):
    if not externals.exists('h5py'):
        raise SkipTest

    from mvpa2.base.hdf5 import h5save, h5load

    ds = _get_test_dataset(include_nodes)

    h5save(fn, ds)
    ds2 = h5load(fn)

    assert_datasets_equal(ds, ds2)
