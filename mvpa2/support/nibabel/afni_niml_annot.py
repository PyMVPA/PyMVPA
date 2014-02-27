# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Experimental support for AFNI NIML annotation files 

Created on Feb 19, 2012

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)
'''

import numpy as np

from mvpa2.support.nibabel import afni_niml as niml
from mvpa2.support.nibabel import afni_niml_dset as dset

import os

def rawniml2annot(p):
    '''Converts raw NIML to annotation format'''

    if type(p) is list:
        return map(rawniml2annot, p)
    r = dset.rawniml2dset(p)

    for nd in p['nodes']:
        if nd.get('dset_type', None) == 'LabelTableObject':
            r[r'AFNI_labeltable'] = dset.rawniml2dset(nd)

    return r

def annot2rawniml(a):
    '''Converts annotation to raw NIML format'''
    a = a.copy()
    t = a.pop('AFNI_labeltable')
    t['labels'] = ['R', 'G', 'B', 'A', 'key', 'name']
    t['dset_type'] = 'LabelTableObject_data'
    t['node_indices'] = None
    r = dset.dset2rawniml(t).copy()

    _fix_rawniml_table_output(r)

    t = dset.dset2rawniml(a)
    _fix_rawniml_main_output(t)

    # add the table to the nodes
    t['nodes'].insert(2, r)

    return t

def _fix_rawniml_table_output(r):
    colms_tp = niml.find_attribute_node(r, 'atr_name', 'COLMS_TYPE')
    colms_tp['data'] = ('R_col;G_col;B_col;A_col;'
                        'Node_Index_Label;Node_String_Label')

    colms_st = niml.find_attribute_node(r, 'atr_name', 'COLMS_STATSYM')
    colms_st['data'] = "none;none;none;none;none;none"

    dset = niml.find_attribute_node(r, 'name', 'AFNI_dataset')
    dset['name'] = 'AFNI_labeltable'
    dset['dset_type'] = 'LabelTableObject'
    dset['flipped'] = '0'
    dset['Sgn'] = '0'

    table = niml.find_attribute_node(r, 'data_type', 'Node_Bucket_data')
    table['data_type'] = 'LabelTableObject_data'

def _fix_rawniml_main_output(r):
    main = niml.find_attribute_node(r, 'data_type', 'Node_Bucket_data')
    main['data_type'] = 'Node_Label_data'

    idx = niml.find_attribute_node(r, 'data_type', 'Node_Bucket_node_indices')
    idx['data_type'] = 'Node_Label_node_indices'

    #header = niml.find_attribute_node(r, 'dset_type', 'LabelTableObject')
    #header['dset_type'] = 'Node_Label'


def _merge_indices_addition_values(idxs, last_index_function=np.max):
    n = len(idxs)

    if any(np.sum(idx < 0) for idx in idxs):
        raise ValueError("Unexpected negative values")

    nidxs = map(last_index_function, idxs)
    last_indices = np.cumsum(nidxs)

    addition_values = []
    last_index = 0
    for i, idx in enumerate(idxs):
        addition_values.append(last_index)
        last_index += last_indices[i] + 1

    return addition_values


def merge(annots):
    '''Merges multiple annotations. One use case is merging two hemispheres'''
    n = len(annots)

    def annot2idx_table_data(annot):
        return (annot['node_indices'],
                annot['AFNI_labeltable']['data'],
                annot['data'])

    idxs, tables, datas = map(list, zip(*map(annot2idx_table_data, annots)))

    to_add_idx = _merge_indices_addition_values(idxs)
    idx = np.vstack(idxs[i] + to_add_idx[i] for i in xrange(n))

    # join the table
    ncols = len(tables[0])
    table = []
    for i in xrange(ncols):
        columns = [d[i] for d in tables]

        if all(isinstance(d[i], np.ndarray) and \
                    np.issubdtype(m.dtype, np.int) for m in columns):
            to_add_table = _merge_indices_addition_values(columns)

            for j in xrange(n):
                columns[j] = columns[j] + to_add_table[j]

        table.append(np.hstack(columns))

    data = np.vstack([datas[i] + to_add_table[i] for i in xrange(n)])

    output = annots[0].copy()
    output['node_indices'] = idx
    output['AFNI_labeltable']['data'] = table
    output['data'] = data

    return output


def read(fn, itemifsingletonlist=True):
    return niml.read(fn, itemifsingletonlist, rawniml2annot)

def write(fnout, niml_annot):
    fn = os.path.split(fnout)[1]

    if not type(fn) is str:
        if not isinstance(fnout, basestring):
            raise ValueError("Filename %s should be string" % str)
        fn = str(fn) # ensure that unicode is converted to string

    niml_annot['filename'] = fn
    form = 'text'
    niml.write(fnout, niml_annot, form, annot2rawniml)
