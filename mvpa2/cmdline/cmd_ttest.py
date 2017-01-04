# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""t-test across multiple volumes against some chance level

This is a rudimentary way to perform fixed-effect significance test
across subjects on e.g. searchlight results maps.

"""

# magic line for manpage summary
# man: -*- % simple t-test routine

__docformat__ = 'restructuredtext'

import numpy as np
import copy
import sys
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.base.hdf5 import h5load, h5save
from mvpa2.datasets import vstack
from mvpa2.datasets.base import Dataset

if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import parser_add_common_opt
from mvpa2.datasets.mri import map2nifti, fmri_dataset
from mvpa2.misc.stats import ttest_1samp
import nibabel as nib
import scipy.stats as stats

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

def setup_parser(parser):
    """Setup parser for command line arguments"""
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts
    parser_add_common_opt(parser, 'multidata', required=True)
    parser_add_common_opt(parser, 'mask', required=False)
    parser_add_common_opt(parser, 'output_file', required=True)

    parser.add_argument('-c', '--chance-level', default=0, type=float,
                        help="""chance level performance""")

    parser.add_argument('-s', '--stat', choices=['t', 'z', 'p'],
                        default='t',
                        help="""Store corresponding statistic, e.g. z-value
                        corresponding to the original t-value""")

    parser.add_argument('-a', '--alternative',
                        choices=['greater', 'less', 'two-sided'], default='greater',
                        help="""Which tail of the distribution 'interesting' values
                        belong to.  E.g. if values are accuracies, it would be the
                        'greater', if errors -- the 'less'""")

    parser.add_argument('--isample',
                        default=0, type=int, help="""In case of multi sample dataset,
                        which sample to extract to run the ttest on""")

def guess_backend(fn):
    """Guess which filetype we're dealing with"""
    if fn.endswith('.gz'):
        fn = fn.strip('.gz')
    if fn.endswith('.nii'):
        filetype = 'nifti'
    elif fn.endswith('.hdf5') or fn.endswith('.h5'):
        filetype = 'hdf5'
    else:
        # assume default
        filetype ='nifti'

    return filetype


def run(args):
    """Run it"""
    verbose(1, "Loading %d result files" % len(args.data))

    filetype_in = guess_backend(args.data[0])

    if filetype_in == 'nifti':
        dss = [fmri_dataset(f) for f in args.data]
    elif filetype_in == 'hdf5':
        dss = [h5load(f) for f in args.data]
    data = np.asarray([d.samples[args.isample] for d in dss])

    if args.mask:
        filetype_mask = guess_backend(args.mask)
        if filetype_mask == 'nifti':
            mask = nib.load(args.mask).get_data()
        elif filetype_mask == 'hdf5':
            mask = h5load(args.mask).samples
        out_of_mask = mask == 0
    else:
        # just take where no voxel had a value
        out_of_mask = np.sum(data != 0, axis=0)==0

    t, p = ttest_1samp(data, popmean=args.chance_level, axis=0,
                       alternative=args.alternative)

    if args.stat == 'z':
        if args.alternative == 'two-sided':
            s = stats.norm.isf(p/2)
        else:
            s = stats.norm.isf(p)
        # take the sign of the original t
        s = np.abs(s) * np.sign(t)
    elif args.stat == 'p':
        s = p
    elif args.stat == 't':
        s = t
    else:
        raise ValueError('WTF you gave me? have no clue about %r' % (args.stat,))

    if s.shape != out_of_mask.shape:
        try:
            out_of_mask = out_of_mask.reshape(s.shape)
        except ValueError:
            raise ValueError('Cannot use mask of shape {0} with '
                             'data of shape {1}'.format(out_of_mask.shape, s.shape))
    s[out_of_mask] = 0

    verbose(1, "Saving to %s" % args.output)
    filetype_out = guess_backend(args.output)
    if filetype_out == 'nifti':
        map2nifti(dss[0], data=s).to_filename(args.output)
    else:  # filetype_out is hdf5
        s = Dataset(np.atleast_2d(s), fa=dss[0].fa, a=dss[0].a)
        h5save(args.output, s)
    return s
