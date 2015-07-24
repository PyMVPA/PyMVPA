# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrapper around FSLs halfcosbasis to generate HRF kernels"""

__docformat__ = 'restructuredtext'

import os
from os.path import join as pathjoin
import tempfile
import shutil
import numpy as np
import math

##REF: Name was automagically refactored
def make_flobs(pre=0, rise=5, fall=5, undershoot=5, undershootamp=0.3, 
              nsamples=1, resolution=0.05, nsecs=-1, nbasisfns=2):
    """Wrapper around the FSL tool halfcosbasis.

    This function uses halfcosbasis to generate samples of HRF kernels.
    Kernel parameters can be modified analogous to the Make_flobs GUI
    which is part of FSL. 

    ::

       ^         /-\\
       |        /   \\
       1       /     \\
       |      /       \\
       |     /         \\
       |    /           \\
      -----/             \\     /-----  |
                          \\--/         |  undershootamp
      |    |      |     |        |
      |    |      |     |        |

       pre   rise  fall  undershoot

    Parameters 'pre', 'rise', 'fall', 'undershoot' and 'undershootamp'
    can be specified as 2-tuples (min-max range for sampling) and single 
    value (setting exact values -- no sampling).

    If 'nsec' is negative, the length of the samples is determined 
    automatically to include the whole kernel function (until it returns
    to baseline). 'nsec' has to be an integer value and is set to the next 
    greater integer value if it is not.

    All parameters except for 'nsamples' and 'nbasisfns' are in seconds.
    """
    # create tempdir and temporary parameter file
    pfile, pfilename = tempfile.mkstemp('pyflobs')
    wdir = tempfile.mkdtemp('pyflobs')

    # halfcosbasis can only handle >1 samples
    # so we simply compute two and later ignore the other
    if nsamples < 2:
        rnsamples = 2
    else:
        rnsamples = nsamples

    # make range tuples if not supplied
    if not isinstance(pre, tuple):
        pre = (pre, pre)
    if not isinstance(rise, tuple):
        rise = (rise, rise)
    if not isinstance(fall, tuple):
        fall = (fall, fall)
    if not isinstance(undershoot, tuple):
        undershoot = (undershoot, undershoot)
    if not isinstance(undershootamp, tuple):
        undershootamp = (undershootamp, undershootamp)

    # calc minimum length of hrf if not specified
    # looks like it has to be an integer
    if nsecs < 0:
        nsecs = int( math.ceil( pre[1] \
                                + rise[1] \
                                + fall[1] \
                                + undershoot[1] \
                                + resolution ) )
    else:
        nsecs = math.ceil(nsecs)

    # write parameter file
    pfile = os.fdopen( pfile, 'w' )

    pfile.write(str(pre[0]) + ' ' + str(pre[1]) + '\n')
    pfile.write(str(rise[0]) + ' ' + str(rise[1]) + '\n')
    pfile.write(str(fall[0]) + ' ' + str(fall[1]) + '\n')
    pfile.write(str(undershoot[0]) + ' ' + str(undershoot[1]) + '\n')
    pfile.write('0 0\n0 0\n')
    pfile.write(str(undershootamp[0]) + ' ' + str(undershootamp[1]) + '\n')
    pfile.write('0 0\n')

    pfile.close()

    # call halfcosbasis to generate the hrf samples
    tochild, fromchild, childerror = os.popen3('halfcosbasis' 
                   + ' --hf=' + pfilename
                   + ' --nbfs=' + str(nbasisfns)
                   + ' --ns=' + str(nsecs)
                   + ' --logdir=' + pathjoin(wdir, 'out')
                   + ' --nhs=' + str(rnsamples)
                   + ' --res=' + str(resolution) )
    err = childerror.readlines()
    if len(err) > 0:
        print err
        raise RuntimeError, "Problem while running halfcosbasis."

    # read samples from file into an array
    hrfs = np.fromfile( pathjoin( wdir, 'out', 'hrfsamps.txt' ),
                       sep = ' ' )

    # reshape array to get one sample per row and 1d array only
    # for one sample hrf
    hrfs = \
        hrfs.reshape( len(hrfs)/rnsamples, rnsamples).T[:nsamples].squeeze()

    # cleanup working dir (ignore errors)
    shutil.rmtree( wdir, True )
    # remove paramter file
    os.remove( pfilename )

    # and return an array
    return( hrfs )

