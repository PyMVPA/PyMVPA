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

import random, numpy as np, os, time, sys, socket
from mvpa2.support.nibabel import afni_niml_types as types
from mvpa2.support.nibabel import afni_niml as niml

from mvpa2.base import warning, debug

def _string2list(s, SEP=";", warn_if_no_string=True):
    '''splits a string by SEP; if the last element is empty then it is not returned

    The rationale is that AFNI/NIML like to close the string with a ';' which
    would return one (empty) value too many

    If the input is already a list that has lists or tuples,
    by default a warning is thrown because SUMA may puke over it.
    '''
    if isinstance(s, (list, tuple)):
        if warn_if_no_string and \
                    any(isinstance(s_elem, (list, tuple)) for s_elem in s):
            # a short representation of the offending structure
            s_str = '%r' % s if len(s) <= 1 else '[%s ... %s]' % (s[0], s[-1])
            warning('Two-dimensional string structure %s found - '
                        'it may not be readable by SUMA.' % s_str)
        return s

    r = s.split(SEP)
    if not r[-1]:
        r = r[:-1]
    return r

def rawniml2dset(p):
    if type(p) is list:
        return map(rawniml2dset, p)

    assert type(p) is dict and all([f in p for f in ['dset_type', 'nodes']]), p
    #assert type(p) is dict and all([f in p for f in ['nodes']]), p

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
            r[name] = data
            #raise ValueError("Unexpected node %s" % name)

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

def _dset_nrows_ncols(s):
    if type(s) is dict and 'data' in s:
        data = s['data']
    else:
        data = s

    if isinstance(data, np.ndarray):
        sh = s['data'].shape
        nrows = sh[0]
        ncols = 1 if len(sh) == 1 else sh[1]
    elif isinstance(data, list):
        lengths = set([len(d) for d in data])
        if len(lengths) != 1:
            raise ValueError('nonmatching lengths', lengths)
        nrows = lengths.pop()
        ncols = len(data)
    else:
        raise ValueError('not understood: %s' % data)

    return nrows, ncols

def _dset2rawniml_nodeidxs(s):
    nrows, _ = _dset_nrows_ncols(s)

    node_idxs = s.get('node_indices') if 'node_indices' in s else np.arange(nrows, dtype=np.int32)

    if not node_idxs is None:
        if not type(node_idxs) is np.ndarray:
            node_idxs = np.asarray(node_idxs, dtype=np.int32)

        if node_idxs.size != nrows:
            raise ValueError("Size mismatch for node indices (%r) and data (%r)" %
                             (node_idxs.size, nrows))

        if node_idxs.shape != (nrows, 1):
            node_idxs = np.reshape(node_idxs, ((nrows, 1))) # reshape to column vector if necessary

    def is_sorted(v): # O(1) in best case and O(n) in worst case (unlike sorted())
        if v is None:
            return None
        n = len(v)
        return n == 0 or all(v[i] <= v[i + 1] for i in xrange(n - 1))

    return dict(data_type='Node_Bucket_node_indices',
                name='INDEX_LIST',
                data=node_idxs,
                sorted_node_def='Yes' if is_sorted(node_idxs) else 'No')

def _dset2rawniml_datarange(s):
    data = s['data']

    try:
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
    except:
        return dict(atr_name='COLMS_RANGE',
                    data=None)

def _dset2rawniml_labels(s):
    _, ncols = _dset_nrows_ncols(s)

    labels = s.get('labels', None)
    if labels is None:
        labels = ['col_%d' % i for i in xrange(ncols)]
    elif type(labels) != list:
        labels = list(labels)
    if len(labels) != ncols:
        raise ValueError("Wrong number of labels (%s): found %d but expected %d" %
                         (labels, len(labels), ncols))
    return dict(atr_name='COLMS_LABS',
                data=labels)

def _dset2rawniml_history(s):
    try:
        logprefix = ('[%s@%s: %s]' % (os.environ.get('USER', 'UNKNOWN'),
                                      socket.gethostname(),
                                      time.asctime()))
    except:
        logprefix = ''
    # history
    history = s.get('history', '')
    if history and not history.endswith('\n'):
        history += ('\n')
    history += '%s Saved by %s:%s' % (logprefix,
                                    __file__,
                                    sys._getframe().f_code.co_name)

    return dict(atr_name='HISTORY_NOTE',
                data=history)

def _dset2rawniml_datatypes(s):
    data = s['data']
    _, ncols = _dset_nrows_ncols(s)
    # XXX does not support mixed types
    datatype = ['Generic_Int' if types.numpy_data_isint(data) else 'Generic_Float'] * ncols
    return dict(atr_name='COLMS_TYPE',
                data=datatype)

def _dset2rawniml_stats(s):
    data = s['data']
    _, ncols = _dset_nrows_ncols(s)
    stats = s.get('stats', None)

    if stats is None:
        stats = ['none'] * ncols
    return dict(atr_name='COLMS_STATSYM',
                data=stats)

def _dset2rawniml_anything_else(s):
    ignore_keys = ['data', 'stats', 'labels', 'history', 'dset_type', 'node_indices']

    ks = s.keys()
    niml = []
    for k in ks:
        if k in ignore_keys:
            continue
        niml_elem = dict(data=s[k], name=k)

        try:
            niml.append(_dset2rawniml_complete(niml_elem))
        except TypeError:
            debug('NIML', 'Warning: unable to convert value for key %s' % k)

    return niml

def _dset2rawniml_complete(r):
    '''adds any missing information and ensures data is formatted properly'''

    # if data is a list of strings, join it and store it as a string
    # otherwise leave data untouched
    if types.numpy_data_isstring(r['data']):
        r['data'] = list(r['data'])

    while True:
        data = r['data']
        tp = type(data)

        if types.numpy_data_isstring(r['data']):
            r['data'] = list(r['data'])

        elif tp is list:
            if all(isinstance(d, basestring)  for d in data):
                r['data'] = ";".join(data)
            else:
                tp = 'mixed'
                break

        else:
            break # we're done

    if tp == 'mixed':
        #data = [types.nimldataassupporteddtype(d) for d in data]
        #r['data'] = data

        nrows, ncols = _dset_nrows_ncols(data)
        r['ni_dimen'] = str(nrows)
        tpstrs = []
        for d in data:
            if isinstance(d, basestring) or \
                    (type(d) is list and
                            all(isinstance(di, basestring) for di in d)):
                tpstr = 'String'
            elif isinstance(d, np.ndarray):
                tpstr = types.numpy_type2name(d.dtype)
            else:
                raise ValueError('unrecognized type %s' % type(d))
            tpstrs.append(tpstr)
        r['ni_type'] = ','.join(tpstrs)

    elif issubclass(tp, basestring):
        r['ni_dimen'] = '1'
        r['ni_type'] = 'String'
    elif tp is np.ndarray:
        data = types.nimldataassupporteddtype(data)
        if len(data.shape) == 1:
            data = np.reshape(data, (data.shape[0], 1))

        r['data'] = data # ensure we store a supported type


        nrows, ncols = data.shape
        r['ni_dimen'] = str(nrows)
        tpstr = types.numpy_type2name(data.dtype)
        r['ni_type'] = '%d*%s' % (ncols, tpstr) if nrows > 1 else tpstr
    elif not data is None:
        raise TypeError('Illegal type %r in %r' % (tp, data))

    if not 'name' in r:
        r['name'] = 'AFNI_atr'

    return r

def _remove_empty_nodes(nodes):
    tp = type(nodes)
    if tp is list:
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if type(node) is dict and 'data' in node and node['data'] is None:
                nodes.pop(i)
            else:
                i += 1
    elif tp is dict:
        for v in nodes.itervalues():
            _remove_empty_nodes(v)


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
    _remove_empty_nodes(nodes)

    more_nodes = filter(lambda x:not x is None, _dset2rawniml_anything_else(s))

    r['nodes'] = nodes + more_nodes
    return r

def read(fn, itemifsingletonlist=True):
    return niml.read(fn, itemifsingletonlist, rawniml2dset)

def write(fnout, dset, form='binary'):
    fn = os.path.split(fnout)[1]

    if not type(fn) is str:
        if not isinstance(fnout, basestring):
            raise ValueError("Filename %s should be string" % str)
        fn = str(fn) # ensure that unicode is converted to string

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
        afni_niml_dset-like dictionary with at least fields 'data' and
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

def ttest(dsets, sa_labels=None, return_values='mt',
          set_NaN_to=0., compare_to=0.):
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
    set_NaN_to: float or None (default: 0.)
        the value that NaNs in dsets replaced by. If None then NaNs are kept.
    compare_to: float (default: 0.)
        t-tests are compared against the null hypothesis of a mean of
        compare_to.

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
                    sa_labels = dset['labels']
                    dset_labels = sa_labels
                else:
                    sa_labels = range(sh[1])
                    dset_labels = ['%d' % j for j in sa_labels]
            else:
                dset_labels = sa_labels


            nc = len(dset_labels) if dset_labels else sh[1]
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

    # subtract the value it is compared to
    # so that it now tests against a mean of zero
    if do_m:
        m = np.mean(data, axis=2)

    if do_t:
        from scipy import stats
        t = stats.ttest_1samp(data - compare_to, 0., axis=2)[0]

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

