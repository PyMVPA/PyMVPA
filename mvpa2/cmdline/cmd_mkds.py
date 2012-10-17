# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Create a PyMVPA dataset from various sources.

This command converts data from various sources, such as text files, Numpy's
NPY files, and MR (magnetic resonance) images into a PyMVPA dataset that is
stored in HDF5 format. An arbitrary number of sample and feature attributes can
be added to a dataset, and individual attributes can be read from
heterogeneous sources (e.g. they do not have to be all from text files).

For datasets from MR images this command also supports automatic conversion
of additional images into (volumetric) feature attributes. This can be useful
for describing features with, for example, atlas labels.

LOAD DATA FROM TEXT FILES

All options for loading data from text files support optional parameters to
tweak the conversion:

  --sa-txt <mandatory values> [DELIMITER [DTYPE [SKIPROWS [COMMENTS]]]]

where 'DELIMITER' is the string that is used to separate values in the input
file, 'DTYPE' is any identifier of a NumPy data type (e.g. 'int', or 'float32'),
'SKIPROWS' is an integer indicating how many lines at the beginning of the
respective file shall be ignored, and 'COMMENTS' is a string indicating how
to-be-ignored comment lines are prefixed in the file.

LOAD DATA FROM NUMPY NPY FILES

All options for loading data from NumPy NPY files support an optional parameter:

  --fa-npy <mandatory values> [MEMMAP]

where 'MEMMAP' is a flag  that triggers whether the respective file shall be
read by memory-mapping, i.e. not read (immediately) into memory. Enable with
'yes', '1', 'true', 'enable' or 'on'.

Examples:

Load 4D MRI image, assign atlas labels to a feature attribute, and attach class
labels from a text file. The resulting dataset is stored as 'ds.hdf5' in
the current directory.

  $ pymvpa2 mkds -o ds --from-mri bold.nii.gz --vol-attr area harvox.nii.gz --sa-txt targets labels.txt

"""

# magic line for manpage summary
# man: -*- % create a PyMVPA dataset from various sources

__docformat__ = 'restructuredtext'

import numpy as np
import argparse
from mvpa2.base.hdf5 import h5save
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
        import parser_add_common_args, parser_add_common_opt, arg2bool

def _load_from_txt(args):
    defaults = dict(dtype=None, delimiter=None, skiprows=0, comments=None)
    if len(args) > 1:
        defaults['delimiter'] = args[1]
    if len(args) > 2:
        defaults['dtype'] = args[2]
    if len(args) > 3:
        defaults['skiprows'] = int(args[3])
    if len(args) > 4:
        defaults['comments'] = args[4]
    data = np.loadtxt(args[0], **defaults)
    return data

def _load_from_npy(args):
    defaults = dict(mmap_mode=None)
    if len(args) > 1 and arg2bool(args[1]):
        defaults['mmap_mode'] = 'r'
    data = np.load(args[0], **defaults)
    return data


parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

txtsrc_args = ('options for input from text file', [
    ('--from-txt', dict(type=str, nargs='+', metavar='VALUE',
        help="""load samples from a text file. The first value
                is the filename the data will be loaded from. Additional
                values modifying the way the data is loaded are described in the
                section "Load data from text files".""")),
    ('--sa-txt', dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""load sample attribute from a text file. The first value
                is the desired attribute name, the second value is the filename
                the attribute will be loaded from. Additional values modifying
                the way the data is loaded are described in the section
                "Load data from text files".""")),
    ('--fa-txt', dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""load feature attribute from a text file. The first value
                is the desired attribute name, the second value is the filename
                the attribute will be loaded from. Additional values modifying
                the way the data is loaded are described in the section
                "Load data from text files".""")),
    ('--sa-attr', dict(type=str, metavar='FILENAME',
        help="""load sample attribute values from an legacy 'attributes file'.
                Column data is read as "literal". Only two column files
                ('targets' + 'chunks') without headers are supported. This
                option allows for reading attributes files from early PyMVPA
                versions.""")),
])

numpysrc_args = ('options for input from Numpy array', [
    ('--from-npy', dict(type=str, nargs='+', metavar='VALUE',
        help="""load samples from a Numpy .npy file. Compressed files (i.e.
             .npy.gz) are supported as well. The first value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
    ('--sa-npy', dict(type=str, nargs='+', metavar='VALUE', action='append',
        help="""load sample attribute from a Numpy .npy file. Compressed files
             (i.e. .npy.gz) are supported as well. The first value is the
             desired attribute name, the second value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
    ('--fa-npy', dict(type=str, nargs='+', metavar='VALUE', action='append',
        help="""load feature attribute from a Numpy .npy file. Compressed files
             (i.e. .npy.gz) are supported as well. The first value is the
             desired attribute name, the second value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
])

mrisrc_args = ('options for input from MR images', [
    ('--from-mri', {
        'type': str,
        'nargs': '+',
        'metavar': 'IMAGE',
        'help': """load data from an MR image, such as a NIfTI file. This can
                either be a single 4D image, or a list of 3D images, or a
                combination of both."""}),
    ('--mask', {
        'type': str,
        'metavar': 'IMAGE',
        'help': """mask image file with the same dimensions as an input data
                sample. All voxels corresponding to non-zero mask elements will
                be permitted into the dataset."""}),
    ('--vol-attr', {
        'type': str,
        'nargs': 2,
        'action': 'append',
        'metavar': 'ARG',
        'help': """attribute name (1st argument) and image file with the same
                dimensions as an input data sample (2nd argument). The image
                data will be added as a feature attribute under the specified
                name."""}),
])


def setup_parser(parser):
    # order of calls is relevant!
    inputsrcsgrp = parser.add_argument_group('input data sources')
    srctypesgrp = inputsrcsgrp.add_mutually_exclusive_group()
    for src in (txtsrc_args, numpysrc_args, mrisrc_args):
        srcgrp = parser.add_argument_group(src[0])
        # make sure the main src arg is exclusive
        srctypesgrp.add_argument(src[1][0][0], ** src[1][0][1])
        # add the rest
        for opt in src[1][1:]:
            srcgrp.add_argument(opt[0], **opt[1])
    outputgrp = parser.add_argument_group('output options')
    parser_add_common_opt(outputgrp, 'output_file', required=True)
    parser_add_common_args(outputgrp, opt=['hdf5compression'])

def run(args):
    if not args.from_txt is None:
        verbose(1, "Load data from TXT file '%s'" % args.from_txt)
        samples = _load_from_txt(args.from_txt)
        ds = Dataset(samples)
    elif not args.from_npy is None:
        verbose(1, "Load data from NPY file '%s'" % args.from_npy)
        samples = _load_from_npy(args.from_npy)
        ds = Dataset(samples)
    elif not args.from_mri is None:
        verbose(1, "Load data from MRI image(s) %s" % args.from_mri)
        from mvpa2.datasets.mri import fmri_dataset
        vol_attr = dict()
        if not args.vol_attr is None:
            vol_attr = dict(args.vol_attr)
            if not len(args.vol_attr) == len(vol_attr):
                warning("--vol-attr option with duplicate attribute name: "
                        "check arguments!")
            verbose(2, "Add volumetric feature attributes: %s" % vol_attr)
        ds = fmri_dataset(args.from_mri, mask=args.mask, add_fa=vol_attr)
    # legacy support
    if not args.sa_attr is None:
        from mvpa2.misc.io.base import SampleAttributes
        smpl_attrs = SampleAttributes(args.sa_attr)
        for a in ('targets', 'chunks'):
            ds.sa[a] = getattr(smpl_attrs, a)
    # loop over all attribute configurations that we know
    attr_cfgs = (# var, dst_collection, loader
            ('--sa-txt', args.sa_txt, ds.sa, _load_from_txt),
            ('--fa-txt', args.fa_txt, ds.fa, _load_from_txt),
            ('--sa-npy', args.sa_npy, ds.sa, _load_from_npy),
            ('--fa-npy', args.fa_npy, ds.fa, _load_from_npy),
        )
    for varid, srcvar, dst_collection, loader in attr_cfgs:
        if not srcvar is None:
            for spec in srcvar:
                attr_name = spec[0]
                if not len(spec) > 1:
                    raise argparse.ArgumentTypeError(
                        "%s option need at least two values " % varid +
                        "(attribute name and source filename (got: %s)" % spec)
                if dst_collection is ds.sa:
                    verbose(2, "Add sample attribute '%s' from '%s'"
                               % (attr_name, spec[1]))
                else:
                    verbose(2, "Add feature attribute '%s' from '%s'"
                               % (attr_name, spec[1]))
                attr = loader(spec[1:])
                try:
                    dst_collection[attr_name] = attr
                except ValueError, e:
                    # try making the exception more readable
                    e_str = str(e)
                    if e_str.startswith('Collectable'):
                        raise ValueError('attribute %s' % e_str[12:])
                    else:
                        raise e
    verbose(2, "Dataset summary %s" % (ds.summary()))
    # and store
    outfilename = args.output
    if not outfilename.endswith('.hdf5'):
        outfilename += '.hdf5'
    verbose(1, "Save dataset to '%s'" % outfilename)
    h5save(outfilename, ds, mkdir=True, compression=args.compression)
