# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
AFNI NIML ROI (region of interest) read support

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)

'''

import numpy as np
from mvpa2.support.nibabel import afni_niml

def read(fn):
    '''Reads a NIML ROI file (typical extension .niml.roi)
    
    Parameters
    ----------
    fn: str
        filename of NIML ROI file
    
    Returns
    -------
    rois: list of dict
        A list of ROIs found in fn. Each element represents a single ROI and
        is a dictionary d with keys from the header information. In addition
        it contains d['edges'], a list of numpy arrays with the node indices
        for each edge, and d['areas'], a list of numpy arrays with the node
        indices of each node
    '''

    with open(fn) as f:
        lines = f.read().split('\n')

    # some headers should be converted 
    color2array = lambda x:np.asarray(map(float, x.split()))
    header_convertors = dict(iLabel=int,
                             Type=int,
                             FillColor=color2array,
                             EdgeColor=color2array,
                             EdgeThickness=int,
                             ni_dimen=int)

    rois = []
    for line in lines:
        if not line:
            continue
        elif line.startswith('#'):

            # process header line
            if line.startswith('# <Node_ROI'):
                roi = dict()
                rois.append(roi)
            elif line.startswith('# </Node_ROI>') or line == '# >':
                continue
            elif ' = ' in line:
                k, v = line.lstrip('#').strip().split(' = ')
                v = afni_niml.decode_escape(v.strip('"'))

                if k in header_convertors:
                    f = header_convertors[k]
                    v = f(v)
                roi[k] = v
            else:
                raise ValueError("Illegal line: %s" % line)
        else:
            v = np.asarray(map(int, line.split()), dtype=np.int).ravel()

            tp = v[1]
            n = v[2]
            nodes = v[3:]

            k = {4:'edges', 1:'areas'}[tp]
            if not k in roi:
                roi[k] = []
            roi[k].append(nodes)

    return rois


def niml_roi2roi_mapping(rois):
    '''Converts NIML ROI representation in mapping from ROI labels 
    to node indices
    
    Parameters
    ----------
    roi: list of dictionaries
        NIML ROIs representation, e.g. from read()
    
    Returns
    -------
    roi_mapping: dict
        A mapping from ROI labels to numpy arrays with node indices 
        
    Notes
    -----
    It is assumed that all labels in the rois are unique, otherwise
    an exception is raised
    '''


    n = len(rois)
    keys = [roi['Label'] for roi in rois]

    if len(set(keys)) != n:
        raise ValueError("Not unique keys in %r" % keys)

    roi_mapping = dict()
    for roi in rois:
        key = roi['Label']
        all_nodes = np.zeros((0,), dtype=np.int)
        for nodes in roi['areas']:
            all_nodes = np.union1d(all_nodes, nodes)
        roi_mapping[key] = all_nodes

    return roi_mapping

def read_mapping(roi):
    '''Converts NIML ROI representation in mapping from ROI labels 
    to node indices
    
    Parameters
    ----------
    roi: list of dictionaries or str
        NIML ROIs representation, e.g. from read(), or a filename
        with ROIs specifications 
    
    Returns
    -------
    roi_mapping: dict
        A mapping from ROI labels to numpy arrays with node indices 
        
    Notes
    -----
    It is assumed that all labels in the rois are unique, otherwise
    an exception is raised
    '''
    return from_any(roi, postproc=niml_roi2roi_mapping)

def from_any(roi, postproc=None):
    '''Returns an ROI representation
    
    Parameters
    ----------
    roi: str or list
        Filename or list of ROI representations
    postproc: callable or None (default: None)
        Postprocessing that is applied after processing the ROIs
    '''

    if type(roi) is list:
        if not all(['Label' in r for r in roi]):
            raise ValueError("Not understood: list %r" % roi)
    elif isinstance(roi, basestring):
        roi = read(roi)

    if not postproc is None:
        roi = postproc(roi)

    return roi
