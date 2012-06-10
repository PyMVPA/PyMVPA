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

        # put in alldata
        alldata[:, 0] = nodeidxs[:, 0]
        alldata[:, 1:] = data[:]
        data = alldata
        fmt = ['%d']
    else:
        fmt = []

    # by default store with 5 decimals precision
    fmt.extend(['%.5f' for _ in xrange(nt)])

    np.savetxt(fnout, data, fmt, ' ')

