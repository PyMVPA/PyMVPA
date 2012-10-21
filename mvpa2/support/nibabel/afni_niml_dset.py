# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
AFNI NIML dataset I/O support.
Usually this type of datasets are used for functional data (timeseries, 
preprocessed data), statistical maps or searchlight results. 

Created on Feb 19, 2012

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)

Files that are read with the afni_niml_dset.read function contain a dict
with the following fields:

   .data           PxN data for P nodes and N columns (values per node).
   .node_indices   P indices of P nodes that data refers to (base 0)
   .history        String with history information
   .stats          list with statistical information for each column.
   .labels         list with labels of the data columns
   .dset_type      String with the data set type 
   
Similarly, such a dict can be saved to a .niml.dset file using the 
afni_niml_dset.write function
'''

import random, numpy as np, os, time, sys
from mvpa2.support.nibabel import afni_niml_types as types
from mvpa2.support.nibabel import afni_niml as niml

def _string2list(s, SEP=";"):
    '''splits a string by SEP; if the last element is empty then it is not returned
    
    The rationale is that AFNI/NIML like to close the string with a ';' which 
    would return one (empty) value too many'''
    r = s.split(SEP)
    if not r[-1]:
        r = r[:-1]
    return r

def rawniml2dset(p):
    if type(p) is list:
        return map(rawniml2dset, p)

    assert type(p) is dict and all([f in p for f in ['dset_type', 'nodes']]), p

    r = dict()
    r['dset_type'] = p['dset_type']

    for node in p['nodes']:
        assert 'name' in node

        name = node['name']
        data = node.get('data', None)

        if name == 'INDEX_LIST':
            r['node_indices'] = data
        elif name == 'SPARSE_DATA':
            r['data'] = data
        elif name == 'AFNI_atr':
            atr = node['atr_name']

            if atr == 'HISTORY_NOTE':
                r['history'] = data
            elif atr == 'COLMS_STATSYM':
                r['stats'] = _string2list(data)
            elif atr == 'COLMS_LABS':
                r['labels'] = _string2list(data)
        else:
            continue; #raise ValueError("Unexpected node %s" % name)

    return r


def _dset2rawniml_header(s):
    r = dict()

    # set dataset type, default is Node_Bucket
    r['dset_type'] = s.get('dset_type', 'Node_Bucket')

    # make a new id code of 24 characters, all uppercase letters
    r['self_idcode'] = niml.getnewidcode()
    r['filename'] = s.get('filename', 'null')
    r['label'] = r['filename']
    r['name'] = 'AFNI_dataset'
    r['ni_form'] = 'ni_group'

    return r

def _dset2rawniml_data(s):
    return dict(data_type='Node_Bucket_data',
                name='SPARSE_DATA',
                data=s['data'])

def _dset2rawniml_nodeidxs(s):
    nrows = s['data'].shape[0]

    node_idxs = s.get('node_indices') if 'node_indices' in s else np.arange(nrows, dtype=np.int32)
    if not type(node_idxs) is np.ndarray:
        node_idxs = np.asarray(node_idxs, dtype=np.int32)

    if node_idxs.size != nrows:
        raise ValueError("Size mismatch for node indices (%r) and data (%r)" %
                         (node_idxs.size, nrows))

    if node_idxs.shape != (nrows, 1):
        node_idxs = np.reshape(node_idxs, ((nrows, 1))) # reshape to column vector if necessary

    def is_sorted(v): # O(1) in best case and O(n) in worst case (unlike sorted())
        n = len(v)
        return n == 0 or all(v[i] <= v[i + 1] for i in xrange(n - 1))

    return dict(data_type='Node_Bucket_node_indices',
                name='INDEX_LIST',
                data=node_idxs,
                sorted_node_def='Yes' if is_sorted(node_idxs) else 'No')

def _dset2rawniml_datarange(s):
    data = s['data']

    minpos = np.argmin(data, axis=0)
    maxpos = np.argmax(data, axis=0)

    f = types.numpy_data2printer(data) # formatter function
    r = []
    for i in xrange(len(minpos)):
        mnpos = minpos[i]
        mxpos = maxpos[i]
        r.append('%s %s %d %d' % (f(data[mnpos, i]), f(data[mxpos, i]), mnpos, mxpos))

    # range of data in each column
    return dict(atr_name='COLMS_RANGE',
                data=r)

def _dset2rawniml_labels(s):
    ncols = s['data'].shape[1]
    labels = list(s.get('labels', None) or ('col_%d' % i for i in xrange(ncols)))
    if len(labels) != ncols:
        raise ValueError("Wrong number of labels: found %d but expected %d" %
                         (len(labels, ncols)))
    return dict(atr_name='COLMS_LABS',
                data=labels)

def _dset2rawniml_history(s):
    logprefix = ('[%s@%s: %s]' % (os.environ['USER'],
                                os.uname()[1],
                                time.asctime()))
    # history
    history = s.get('history', '')
    if history and not history.endswith('\n'):
        history += ('\n')
    history += '%s Saved by %s:%s' % (logprefix,
                                    __file__,
                                    sys._getframe().f_code.co_name)
    history = str(history.decode('utf-8'))
    return dict(atr_name='HISTORY_NOTE',
                data=history)

def _dset2rawniml_datatypes(s):
    data = s['data']
    ncols = data.shape[1]
    datatype = ['Generic_Int' if types.numpy_data_isint(data) else 'Generic_Float'] * ncols
    return dict(atr_name='COLMS_TYPE',
                data=datatype)

def _dset2rawniml_stats(s):
    data = s['data']
    ncols = data.shape[1]
    stats = s.get('stats', None) or ['none'] * ncols
    return dict(atr_name='COLMS_STATSYM',
                data=stats)

def _dset2rawniml_complete(r):
    '''adds any missing information and ensures data is formatted properly'''

    # if data is a list of strings, join it and store it as a string
    # otherwise leave data untouched
    while True:
        data = r['data']
        tp = type(data)

        if tp is list:
            if not data or type(data[0]) is str:
                r['data'] = ";".join(data)
                # new data and tp values are set in next (and final) iteration
            else:
                raise TypeError("Illegal type %r" % tp)
        else:
            break # we're done

    if tp is str:
        r['ni_dimen'] = '1'
        r['ni_type'] = 'String'
    elif tp is np.ndarray:
        data = types.nimldataassupporteddtype(data)
        r['data'] = data # ensure we store a supported type

        nrows, ncols = data.shape
        r['ni_dimen'] = str(nrows)
        tpstr = types.numpy_type2name(data.dtype)
        r['ni_type'] = '%d*%s' % (ncols, tpstr) if nrows > 1 else tpstr
    else:
        raise TypeError('Illegal type %r' % tp)

    if not 'name' in r:
        r['name'] = 'AFNI_atr'

    return r


def dset2rawniml(s):
    if type(s) is list:
        return map(dset2rawniml, s)
    elif type(s) is np.ndarray:
        s = dict(data=s)

    if not 'data' in s:
        raise ValueError('No data?')

    r = _dset2rawniml_header(s)
    builders = [_dset2rawniml_data,
              _dset2rawniml_nodeidxs,
              _dset2rawniml_labels,
              _dset2rawniml_datarange,
              _dset2rawniml_history,
              _dset2rawniml_datatypes,
              _dset2rawniml_stats]

    nodes = [_dset2rawniml_complete(build(s)) for build in builders]
    r['nodes'] = nodes
    return r

def read(fn, itemifsingletonlist=True):
    return niml.read(fn, itemifsingletonlist, rawniml2dset)

def write(fnout, dset, form='binary'):
    fn = os.path.split(fnout)[1]
    dset['filename'] = fn
    niml.write(fnout, dset, form, dset2rawniml)

def sparse2full(dset, pad_to_ico_ld=None, pad_to_node=None,
                ico_ld_surface_count=1, set_missing_values=0):
    '''
    Creates a 'full' dataset which has values associated with all nodes
    
    Parameters
    ----------
    dset: dict
        afni_niml_dset-like dictionary with at least a field 'data'
    pad_to_node_ico_ld: int
        number of linear divisions (only applicable if used through 
        AFNI's MapIcosehedron) of the surface this dataset refers to. 
    pad_to_node: int
        number of nodes of the surface this data
    ico_ld_surface_count: int (default: 1)
        if pad_to_ico_ld is set, this sets the number of surfaces that
        were origingally used. The typical use case is using a 'merged'
        surface originally based on a left and right hemisphere
    set_missing_values: int or float (default: 0)
        value to which nodes not present in dset are set.
    
    Returns
    -------
    dset: dict
        afni_niml_dset-like dictionaryu with at least fields 'data' and 
        'node_indices'.
    '''

    if not pad_to_ico_ld is None:
        if pad_to_node:
            raise ValueError("Cannot have both ico_ld and pad_to_node")
        pad_to_node = ico_ld_surface_count * (pad_to_ico_ld ** 2 * 10 + 2)
    else:
        if pad_to_node is None:
            raise ValueError("Need either pad_to_ico_ld or pad_to_node")

    data = dset['data']
    nrows, ncols = data.shape

    node_indices = dset.get('node_indices', np.reshape(np.arange(nrows),
                                                            (-1, 1)))

    # a few sanity checks
    n = len(node_indices)

    if nrows != n:
        raise ValueError('element count mismatch between data (%d) and '
                         'node indices (%d)' % (nrows, n))

    if n > pad_to_node:
        raise ValueError('data has more rows (%d) than pad_to_node (%d)',
                                                (n, pad_to_node))

    full_node_indices_vec = np.arange(pad_to_node)
    full_node_indices = np.reshape(full_node_indices_vec, (pad_to_node, 1))

    full_data = np.zeros((pad_to_node, ncols), dtype=data.dtype) + \
                                                        set_missing_values
    full_data[np.reshape(node_indices, (n,)), :] = data[:, :]

    fulldset = dict(dset) # make a (superficial) copy
    fulldset['data'] = full_data
    fulldset['node_indices'] = full_node_indices

    return fulldset

def from_any(s, itemifsingletonlist=True):
    if isinstance(s, dict) and 'data' in s:
        return s
    elif isinstance(s, basestring):
        return read(s, itemifsingletonlist)
    elif isinstance(s, np.ndarray):
        return dict(data=s)
    else:
        raise ValueError('not recognized input: %r' % s)


def label2index(dset, label):
    if type(label) is list:
        return [label2index(dset, x) for x in label]

    if type(label) is int:
        sh = dset['data'].shape
        if label < 0 or label >= sh[1]:
            raise ValueError('label index %d out of bounds (0.. %d)' %
                                    (label, sh[1]))
        return label

    labels = dset.get('labels', None)
    if labels is None:
        raise ValueError('No labels found')

    for i, k in enumerate(labels):
        if k == label:
            return i

    return None

def ttest(dsets, sa_labels=None, return_values='mt', set_NaN_to=0.):
    '''Runs a one-sample t-test across datasets
    
    Parameters
    ----------
    dsets: str or list of dicts
        (filenames of) NIML dsets, each referring to PxQ data for
        P nodes (features) and Q values per node (samples)
    sa_labels: list of (int or str)
        indices or labels of columns to compare
    return_values: str (default: 'mt')
        'm' or 't' or 'mt' to return sample mean, t-value, or both
    
    Returns
    -------
    dset: dict
        NIML dset-compatible dict with fields 'data', 'labels',
        'stats' and 'node_indices' set.
    '''

    do_m = 'm' in return_values
    do_t = 't' in return_values

    if not (do_m or do_t):
        raise ValueError("Have to return at least m or t")

    ns = len(dsets)

    for i, dset in enumerate(dsets):
        dset = from_any(dset)
        dset_data = dset['data']
        if i == 0:
            sh = dset_data.shape
            if sa_labels is None:
                if 'labels' in dset:
                    sa_labels = sa_labels
                    dset_labels = sa_labels
                else:
                    sa_labels = range(sh[1])
                    dset_labels = ['%d' % j for j in sa_labels]
            else:
                dset_labels = sa_labels

            nc = len(sa_labels)
            nn = sh[0]

            data = np.zeros((nn, nc, ns), dset_data.dtype) # number of nodes, columns, subjects

        if 'node_indices' in dset:
            node_idxs = np.reshape(dset['node_indices'], (-1,))
        else:
            node_idxs = np.arange(nn)

        if i == 0:
            node_idxs0 = node_idxs
        else:
            if set(node_idxs0) != set(node_idxs):
                raise ValueError("non-matching node indices for %d and %d" %
                                    (0, i))

        col_idxs = np.asarray(label2index(dset, sa_labels))
        data[node_idxs, :, i] = dset_data[:, col_idxs]



    if do_m:
        m = np.mean(data, axis=2)

    if do_t:
        from scipy import stats
        t = stats.ttest_1samp(data, 0., axis=2)[0]

    if do_m and do_t:
        r = np.zeros((nn, 2 * nc), dtype=m.dtype)
        r[:, np.arange(0, 2 * nc, 2)] = m
        r[:, np.arange(1, 2 * nc, 2)] = t
    elif do_t:
        r = t
    elif do_m:
        r = m

    pf = []
    stats = []
    if do_m:
        pf.append('m')
        stats.append('None')
    if do_t:
        pf.append('t')
        stats.append('Ttest(%d)' % (ns - 1))

    labs = sum([['%s_%s' % (p, lab) for p in pf] for lab in dset_labels], [])
    stats = stats * nc

    if not set_NaN_to is None:
        r[np.logical_not(np.isfinite(r))] = set_NaN_to


    return dict(data=r, labels=labs, stats=stats, node_indices=node_idxs0)

