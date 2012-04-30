'''
Very simple AFNI 1D support

Created on Feb 12, 2012

@author: nick
'''

import numpy as np
import utils

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


if __name__ == '__main__':
    d = '%s/ref/' % utils._get_fingerdata_dir()
    fn = d + '__test.1D'

    data = np.array([[1, 2], [3, 4], [5, 6]])
    d = write(fn, data)
