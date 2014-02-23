# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Create a PyMVPA dataset from various sources.

This command converts data from various sources, such as text files, NumPy's
NPY files, and MR (magnetic resonance) images into a PyMVPA dataset that gets
stored in HDF5 format. An arbitrary number of sample and feature attributes can
be added to a dataset, and individual attributes can be read from
heterogeneous sources (e.g. they do not have to be all from text files).

For datasets from MR images this command also supports automatic conversion
of additional images into (volumetric) feature attributes. This can be useful
for describing features with, for example, atlas labels.

COMPOSE ATTRIBUTES ON THE COMMAND LINE

Options --add-sa and --add-fa  can be used to compose dataset attributes directly on
the command line. The syntax is:

... --add-sa <attribute name> <comma-separated values> [DTYPE]

where the optional 'DTYPE' is any identifier of a NumPy data type (e.g. 'int',
or 'float32'). If no data type is specified the attribute values will be
strings.

If only one attribute value is given, it will copied and assigned to all
entries in the dataset.

LOAD DATA FROM TEXT FILES

All options for loading data from text files support optional parameters to
tweak the conversion:

... --add-sa-txt <mandatory values> [DELIMITER [DTYPE [SKIPROWS [COMMENTS]]]]

where 'DELIMITER' is the string that is used to separate values in the input
file, 'DTYPE' is any identifier of a NumPy data type (e.g. 'int', or 'float32'),
'SKIPROWS' is an integer indicating how many lines at the beginning of the
respective file shall be ignored, and 'COMMENTS' is a string indicating how
to-be-ignored comment lines are prefixed in the file.

LOAD DATA FROM NUMPY NPY FILES

All options for loading data from NumPy NPY files support an optional parameter:

... --add-fa-npy <mandatory values> [MEMMAP]

where 'MEMMAP' is a flag  that triggers whether the respective file shall be
read by memory-mapping, i.e. not read (immediately) into memory. Enable by
with on of: yes|1|true|enable|on'.

Examples:

Load 4D MRI image, assign atlas labels to a feature attribute, and attach class
labels from a text file. The resulting dataset is stored as 'ds.hdf5' in
the current directory.

  $ pymvpa2 mkds -o ds --mri-data bold.nii.gz --vol-attr area harvox.nii.gz --add-sa-txt targets labels.txt

"""

# magic line for manpage summary
# man: -*- % create a PyMVPA dataset from various sources

__docformat__ = 'restructuredtext'

import numpy as np
import argparse

from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers import process_common_dsattr_opts, \
        hdf2ds, parser_add_common_opt
# necessary to enable dataset.summary()
import mvpa2.datasets.miscfx


parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

datasrc_args = ('input data sources', [
    (('--txt-data',), dict(type=str, nargs='+', metavar='VALUE',
        help="""load samples from a text file. The first value
                is the filename the data will be loaded from. Additional
                values modifying the way the data is loaded are described in the
                section "Load data from text files".""")),
    (('--npy-data',), dict(type=str, nargs='+', metavar='VALUE',
        help="""load samples from a Numpy .npy file. Compressed files (i.e.
             .npy.gz) are supported as well. The first value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
    (('--mri-data',), {
        'type': str,
        'nargs': '+',
        'metavar': 'IMAGE',
        'help': """load data from an MR image, such as a NIfTI file. This can
                either be a single 4D image, or a list of 3D images, or a
                combination of both."""}),
])

mri_args = ('options for input from MR images', [
    (('--mask',), {
        'type': str,
        'metavar': 'IMAGE',
        'help': """mask image file with the same dimensions as an input data
                sample. All voxels corresponding to non-zero mask elements will
                be permitted into the dataset."""}),
    (('--add-vol-attr',), {
        'type': str,
        'nargs': 2,
        'action': 'append',
        'metavar': 'ARG',
        'help': """attribute name (1st argument) and image file with the same
                dimensions as an input data sample (2nd argument). The image
                data will be added as a feature attribute under the specified
                name."""}),
    (('--add-fsl-mcpar',), dict(type=str, metavar='FILENAME', help=
                """6-column motion parameter file in FSL's McFlirt format. Six
                additional sample attributes will be created: mc_{x,y,z} and
                mc_rot{1-3}, for translation and rotation estimates
                respectively.""")),
])


def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts, single_required_hdf5output
    # order of calls is relevant!
    parser_add_common_opt(parser, 'multidata', metavar='dataset', nargs='*',
            default=None)
    parser_add_optgroup_from_def(parser, datasrc_args, exclusive=True)
    parser_add_common_attr_opts(parser)
    parser_add_optgroup_from_def(parser, mri_args)
    parser_add_optgroup_from_def(parser, single_required_hdf5output)

def run(args):
    from mvpa2.base.hdf5 import h5save
    ds = None
    if not args.txt_data is None:
        verbose(1, "Load data from TXT file '%s'" % args.txt_data)
        samples = _load_from_txt(args.txt_data)
        ds = Dataset(samples)
    elif not args.npy_data is None:
        verbose(1, "Load data from NPY file '%s'" % args.npy_data)
        samples = _load_from_npy(args.npy_data)
        ds = Dataset(samples)
    elif not args.mri_data is None:
        verbose(1, "Load data from MRI image(s) %s" % args.mri_data)
        from mvpa2.datasets.mri import fmri_dataset
        vol_attr = dict()
        if not args.add_vol_attr is None:
            # XXX add a way to use the mapper of an existing dataset to
            # add a volume attribute without having to load the entire
            # mri data again
            vol_attr = dict(args.add_vol_attr)
            if not len(args.add_vol_attr) == len(vol_attr):
                warning("--vol-attr option with duplicate attribute name: "
                        "check arguments!")
            verbose(2, "Add volumetric feature attributes: %s" % vol_attr)
        ds = fmri_dataset(args.mri_data, mask=args.mask, add_fa=vol_attr)

    if ds is None:
        if args.data is None:
            raise RuntimeError('no data source specific')
        else:
            ds = hdf2ds(args.data)[0]
    else:
        if args.data is not None:
            verbose(1, 'ignoring dataset input in favor of other data source -- remove either one to disambiguate')

    # act on all attribute options
    ds = process_common_dsattr_opts(ds, args)

    if not args.add_fsl_mcpar is None:
        from mvpa2.misc.fsl.base import McFlirtParams
        mc_par = McFlirtParams(args.add_fsl_mcpar)
        for param in mc_par:
            verbose(2, "Add motion regressor as sample attribute '%s'"
                       % ('mc_' + param))
            ds.sa['mc_' + param] = mc_par[param]

    verbose(3, "Dataset summary %s" % (ds.summary()))
    # and store
    outfilename = args.output
    if not outfilename.endswith('.hdf5'):
        outfilename += '.hdf5'
    verbose(1, "Save dataset to '%s'" % outfilename)
    h5save(outfilename, ds, mkdir=True, compression=args.hdf5_compression)
