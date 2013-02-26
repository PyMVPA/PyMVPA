# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Caret binary file support 

@author: nick
'''

import numpy as np, os, datetime

from mvpa2.support.nibabel import surf

def _end_header_position(s):
    '''Finds where the caret header ends'''
    END_HEADER = 'EndHeader\n'
    end_header_pos = s.find(END_HEADER)
    return end_header_pos + len(END_HEADER) if end_header_pos > 0 else None


def read_topology(fn):
    '''Reads Caret .topo file
    
    Parameters
    ----------
    fn: str
        filename
    
    Returns
    -------
    faces: np.ndarray
        Px3 array for P faces
    '''

    with open(fn) as f:
        s = f.read()

    end_header_pos = _end_header_position(s)
    body = s[end_header_pos:]

    if body.startswith('tag-version'):
        body = body[body.find('\n') + 1:]

    nfaces = np.fromstring(body[:4], dtype='>i4')
    faces = np.fromstring(body[4:], dtype='>i4').reshape((-1, 3))

    return faces


def read(fn, topology_fn=None):
    '''Reads Caret .coord file and also (if it exists) the topology file
    
    Parameters
    ----------
    fn: str
        filename of .coord file
    topology_fn: str or None
        filename of .topo file. If None then it is attempted to deduce
        this filename from the header in fn.
    
    Returns
    -------
    s: surf.Surface
        Surface with the nodes as in fn, and the topology form topology_fn
    '''
    with open(fn) as f:
        s = f.read()

    end_header_pos = _end_header_position(s)

    header = s[:end_header_pos]
    body = s[end_header_pos:]

    # read body
    vertices = np.fromstring(body[4:], dtype='>f4').reshape((-1, 3))

    # see if we can find the topology
    faces = None
    if topology_fn is None:
        lines = header.split('\n')

        for line in lines:
            if line.startswith('topo_file'):
                topo_file = line[10:]

                topo_fn = os.path.join(os.path.split(fn)[0], topo_file)
                if os.path.exists(topo_fn):
                    faces = read_topology(topo_fn)
                    break
    else:
        faces = read_topology(topology_fn)

    if faces is None:
        # XXX print a warning?
        # For now no warning as the Surface.__init__ should print something
        faces = np.zeros((0, 3), dtype=np.int32)

    return surf.Surface(vertices, faces)
