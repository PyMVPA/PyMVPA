# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
Very simple AFNI 1D support (only writing, at the moment)

Created on Feb 12, 2012

@author: Nikolaas. N. Oosterhof (nikolaas.oosterhof@unitn.it)
'''

import numpy as np

def write(fnout, data, nodeidxs=None):
    data = np.array(data)
    nv = data.shape[0]
    nt = 1 if data.ndim == 1 else data.shape[1]
    if nodeidxs != None:
        # make space
        alldata = np.zeros((nv, nt + 1))

        # ensure all in good shape
        nodeidxs = np.reshape(np.array(nodeidxs), (-1, 1))
        data = np.reshape(data, (nv, -1))

        # first column for node indices, remaining columns for data
        alldata[:, 0] = nodeidxs[:, 0]
        alldata[:, 1:] = data[:]
        data = alldata
        fmt = ['%d']
    else:
        fmt = []

    # 5 decimal places should be enough for everyone
    fmt.extend(['%.5f' for _ in xrange(nt)])

    np.savetxt(fnout, data, fmt, ' ')

