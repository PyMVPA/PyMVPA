# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

# magic line for manpage summary
# man: -*- % align the features across multiple datasets into a common space

__docformat__ = 'restructuredtext'

import numpy as np
import argparse
from mvpa2.base.hdf5 import h5save
from mvpa2.base import verbose
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
        import strip_from_docstring, parser_add_common_opt, \
               param2arg, ca2arg

from mvpa2.algorithms.hyperalignment import Hyperalignment

parser_args = {
    'description': strip_from_docstring(Hyperalignment.__doc__,
                                        paragraphs=(4,),
                                        sections=(('Examples', 'Notes'))),
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

_supported_cas = {
    'residual_errors': {
        'output_suffix': '_resid_errors.txt',
        },
    'training_residual_errors': {
        'output_suffix': '_resid_errors_train.txt',
        },
}

_output_specs = {
    'commonspace': {
        'output_suffix': '_commonspace',
        'help': 'Store the final common space dataset after completion of level two.'
        },
    'commonspace-image': {
        'help': 'Equivalent to --commonspace, but store as an image (typically in NIfTI format).'
        },
    'store-transformation': {
        'help': 'Store common space transformation mappers for each training dataset.',
        },
}

_supported_parameters = (
    'alpha', 'level2_niter', 'ref_ds', 'zscore_all', 'zscore_common',
)

def _transform_dss(srcs, masks, mappers, args):
    if __debug__:
        debug('CMDLINE', "loading to-be-transformed data from %s" % srcs)
    dss = args2datasets(srcs, masks)
    verbose(1, "Loaded %i to-be-transformed datasets" % len(dss))
    if __debug__:
        debug('CMDLINE', "transform datasets")
    tdss = [ mappers[i].forward(td) for i, td in enumerate(dss)]
    return tdss, dss


def setup_parser(parser):
    # order of calls is relevant!
    inputargs = parser.add_argument_group('input data arguments')
    parser_add_common_opt(inputargs, 'multidata', required=True)
    parser_add_common_opt(inputargs, 'multimask')
    inputargs.add_argument('-t', '--transform', nargs='+', help="""\
Additional datasets for transformation into the common space. The number and
order of these datasets have to match those of the training dataset arguments
as the correspond mapper will be used to transform each individual dataset.
Likewise, the same masking is applied to these datasets. Transformed
datasets are stored in the same format as the input data.""")
    inputargs.add_argument('--transform-images', nargs='+',
        help="Identical to --transform, but stores transformed data as images.")
    algoparms = parser.add_argument_group('algorithm parameters')
    for param in _supported_parameters:
        param2arg(algoparms, Hyperalignment, param)
    outopts = parser.add_argument_group('output options')
    parser_add_common_opt(outopts, 'output_prefix', required=True)
    for oopt in sorted(_output_specs):
        outopts.add_argument('--%s' % oopt, action='store_true',
            help=_output_specs[oopt]['help'])
    for ca in sorted(_supported_cas):
        ca2arg(outopts, Hyperalignment, ca,
               help="\nOutput will be stored into '<PREFIX>%s'"
                    % _supported_cas[ca]['output_suffix'])

def run(args):
    if __debug__:
        debug('CMDLINE', "loading input data from %s" % args.data)
        debug('CMDLINE', "loading input data masks from %s" % args.masks)
    dss = args2datasets(args.data, args.masks)
    verbose(1, "Loaded %i input datasets" % len(dss))
    if __debug__:
        for i, ds in enumerate(dss):
            debug('CMDLINE', "dataset %i: %s" % (i, str(ds)))
    # TODO at this point more check could be done, e.g. ref_ds > len(dss)
    # assemble parameters
    params = dict([(param, getattr(args, param)) for param in _supported_parameters])
    if __debug__:
        debug('CMDLINE', "configured parameters: '%s'" % params)
    # assemble CAs
    enabled_ca = [ca for ca in _supported_cas if getattr(args, ca)]
    if __debug__:
        debug('CMDLINE', "enabled conditional attributes: '%s'" % enabled_ca)
    hyper = Hyperalignment(enable_ca=enabled_ca, **params)
    verbose(1, "Running hyperalignment")
    promappers = hyper(dss)
    verbose(2, "Alignment reference is dataset %i" % hyper.ca.chosen_ref_ds)
    verbose(1, "Writing output")
    # output
    if args.commonspace_image:
        if __debug__:
            debug('CMDLINE', "write commonspace image")
        from mvpa2.datasets.mri import map2nifti
        import nibabel as nb
        ref_ds = hyper.ca.chosen_ref_ds
        img = map2nifti(dss[ref_ds], hyper.commonspace)
        nb.save(img,
                '%s%s.nii.gz' \
                    % (args.output_prefix,
                       _output_specs['commonspace']['output_suffix']))
    # save on memory and remove the training data
    del dss
    if args.commonspace:
        if __debug__:
            debug('CMDLINE', "write commonspace as hdf5")
        h5save('%s%s.hdf5' % (args.output_prefix,
                              _output_specs['commonspace']['output_suffix']),
               hyper.commonspace,
               compression=9)
    for ca in _supported_cas:
        if __debug__:
            debug('CMDLINE', "check conditional attribute: '%s'" % ca)
        if getattr(args, ca):
            if __debug__:
                debug('CMDLINE', "store conditional attribute: '%s'" % ca)
            np.savetxt('%s%s' % (args.output_prefix,
                                 _supported_cas[ca]['output_suffix']),
                       hyper.ca[ca].value.samples)
    if args.store_transformation:
        for i, pm in enumerate(promappers):
            if __debug__:
                debug('CMDLINE', "store mapper %i: %s" % (i, str(pm)))
            h5save('%s%s.hdf5' % (args.output_prefix, '_map%.3i' % i),
                   pm, compression=9)
    if args.transform:
        tdss, dss = _transform_dss(args.transform, args.masks, promappers, args)
        del dss
        verbose(1, "Store transformed datasets")
        for i, td in enumerate(tdss):
            if __debug__:
                debug('CMDLINE', "store transformed data %i: %s" % (i, str(td)))
            h5save('%s%s.hdf5' % (args.output_prefix, '_transformed%.3i' % i),
                   td, compression=9)
    elif args.transform_images:
        from mvpa2.datasets.mri import map2nifti
        import nibabel as nb
        tdss, dss = _transform_dss(args.transform_images, args.masks, promappers,
                                   args)
        verbose(1, "Store transformed datasets as images")
        for i, td in enumerate(tdss):
            if __debug__:
                debug('CMDLINE', "store transformed data %i: %s" % (i, str(td)))
            img = map2nifti(dss[i], td)
            nb.save(img,
                    '%s%s.nii.gz' \
                        % (args.output_prefix, '_transformed%.3i' % i))
