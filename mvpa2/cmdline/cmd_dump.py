# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dump dataset components in various formats

A single arbitrary dataset component (sample data, sample attribute, feature
attribute, dataset attribute) can be selected and exported into another
format. A list of supported formats and their respective capabilities is below.

PLAIN TEXT OUTPUT

1D and 2D numerical data can be export as plain text. In addition lists of
strings are supported as well. Typically data is exported with one element per
line is, except for 2D numerical matrices, where an entire row is written on
a single line (space-separated).

For all unsupported data a warning is issued and a truncated textual description
of the dataset component is given.

HDF5 STORAGE

Arbitrary data (types) can be stored in HDF5 containers. For simple data that is
natively supported by HDF5 a toplevel HDF5 dataset is created and contains all
data. Complex, not natively supported data types are serialized before stored
in HDF5.

NUMPY'S NPY BINARY FILES

This data format is for storing numerical data (with arbitrary number of
dimensions in binary format).

NIFTI FILES

This data format is for (multi-dimensional) spatial images. Input datasets
should have a mapper that can reverse-map the corresponding dataset component
back into the image space.

Examples:

Print a sample attribute

  $ pymvpa2 dump mydata.hdf5 --sa subj

Export the sample data array into NumPy's .npy format

  $ pymvpa2 dump mydata.hdf5 -s -f npy -d mysamples.npy

"""

# magic line for manpage summary
# man: -*- % export dataset components into other (file) formats

__docformat__ = 'restructuredtext'

import numpy as np
import sys
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.base.hdf5 import h5save
from mvpa2.datasets import Dataset, vstack
from mvpa2.datasets.eventrelated import eventrelated_dataset, find_events
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import parser_add_common_args, hdf2ds, \
           hdf5compression

def _check_destination(args):
    if args.destination is None:
        raise ValueError("no output filename given (missing --destination)")

def to_nifti(dumpy, ds, args):
    from mvpa2.datasets.mri import map2nifti
    # TODO allow overriding the nifti header
    # TODO allow overriding the mapper
    nimg = map2nifti(ds, dumpy)
    nimg.to_filename(args.destination)

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

component_grp = ('options for selecting dataset components', [
    (('-s', '--samples'), dict(action='store_true',
        help="""dump the dataset samples.""")),
    (('--sa',), dict(
        help="""name of the sample attribute to be dumped.""")),
    (('--fa',), dict(
        help="""name of the feature attribute to be dumped.""")),
    (('--da',), dict(
        help="""name of the dataset attribute to be dumped.""")),
])

hdf5_grp =('option for HDF5 output', [
    hdf5compression[1:]
])

def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts
    parser_add_common_args(parser, pos=['multidata'])
    parser_add_optgroup_from_def(parser, component_grp, exclusive=True)
    parser.add_argument('-d', '--destination',
                        help="""output destination. If no output destination
                        is given it will be directed to stdout, if permitted by
                        the data format""")
    parser.add_argument('-f', '--format', default='txt',
                        choices=('hdf5', 'nifti', 'npy', 'txt'),
                        help="""output format""")
    parser_add_optgroup_from_def(parser, hdf5_grp)


def run(args):
    dss = hdf2ds(args.data)
    verbose(3, 'Loaded %i dataset(s)' % len(dss))
    ds = vstack(dss)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    # What?
    if args.samples:
        dumpy = ds.samples
    elif not ((args.sa is None) and (args.fa is None) and (args.da is None)):
        for attr, col in ((args.sa, ds.sa), (args.fa, ds.fa), (args.da, ds.a)):
            if attr is None:
                continue
            try:
                dumpy = col[attr].value
            except KeyError:
                raise ValueError("unknown attribute '%s', known are %s)"
                                 % (attr, col.keys()))
    else:
        raise ValueError('no dataset component chosen')
    # How?
    if args.format == 'txt':
        if args.destination:
            out = open(args.destination, 'w')
        else:
            out = sys.stdout
        try:
            # trying to write numerical data
            fmt=None
            if np.issubdtype(dumpy.dtype, int):
                fmt='%i'
            elif np.issubdtype(dumpy.dtype, float):
                fmt='%G'
            if fmt is None:
                np.savetxt(out, dumpy)
            else:
                np.savetxt(out, dumpy, fmt=fmt)
        except:
            # it could be something 1d that we can try to print
            if hasattr(dumpy, 'shape') and len(dumpy.shape) == 1:
                for v in dumpy:
                    print v
            else:
                warning("conversion to plain text is not supported for "
                        "this data type")
                # who knows what it is
                out.write(repr(dumpy))
        if not out is sys.stdout:
            out.close()
    elif args.format == 'hdf5':
        _check_destination(args)
        if not args.destination.endswith('.hdf5'):
            args.destination += '.hdf5'
        h5save(args.destination, dumpy)
    elif args.format == 'npy':
        _check_destination(args)
        np.save(args.destination, dumpy)
    elif args.format == 'nifti':
        _check_destination(args)
        to_nifti(dumpy, ds, args)
    return ds
