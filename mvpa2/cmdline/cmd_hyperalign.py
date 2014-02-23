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
               param2arg, ca2arg, arg2ds

from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.mappers.procrustean import ProcrusteanMapper

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
    'store-transformation': {
        'help': 'Store common space transformation mappers for each training dataset.',
        },
}

_supported_parameters = (
    'alpha', 'level2_niter', 'ref_ds', 'zscore_all', 'zscore_common',
)

def _transform_dss(srcs, mappers, args):
    if __debug__:
        debug('CMDLINE', "loading to-be-transformed data from %s" % srcs)
    dss = [arg2ds(d) for d in srcs]
    verbose(1, "Loaded %i to-be-transformed datasets" % len(dss))
    if __debug__:
        debug('CMDLINE', "transform datasets")
    tdss = [ mappers[i].forward(td) for i, td in enumerate(dss)]
    return tdss, dss


def setup_parser(parser):
    # order of calls is relevant!
    inputargs = parser.add_argument_group('input data arguments')
    parser_add_common_opt(inputargs, 'multidata', action='append',
            required=True)
    parser_add_common_opt(
            inputargs, 'multidata',
            names=('-t', '--transform'), dest='transform',
             help="""\
Additional datasets for transformation into the common space. The number and
order of these datasets have to match those of the training dataset arguments
as the correspond mapper will be used to transform each individual dataset.""")
    algoparms = parser.add_argument_group('algorithm parameters')
    for param in _supported_parameters:
        param2arg(algoparms, Hyperalignment, param)
    outopts = parser.add_argument_group('output options')
    parser_add_common_opt(outopts, 'output_prefix', required=True)
    parser_add_common_opt(outopts, 'hdf5compression')
    for oopt in sorted(_output_specs):
        outopts.add_argument('--%s' % oopt, action='store_true',
            help=_output_specs[oopt]['help'])
    for ca in sorted(_supported_cas):
        ca2arg(outopts, Hyperalignment, ca,
               help="\nOutput will be stored into '<PREFIX>%s'"
                    % _supported_cas[ca]['output_suffix'])

def run(args):
    print args.data
    dss = [arg2ds(d)[:,:100] for d in args.data]
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
    hyper = Hyperalignment(enable_ca=enabled_ca,
                           alignment=ProcrusteanMapper(svd='dgesvd',
                                                       space='commonspace'),
                           **params)
    verbose(1, "Running hyperalignment")
    promappers = hyper(dss)
    verbose(2, "Alignment reference is dataset %i" % hyper.ca.chosen_ref_ds)
    verbose(1, "Writing output")
    # save on memory and remove the training data
    del dss
    if args.commonspace:
        if __debug__:
            debug('CMDLINE', "write commonspace as hdf5")
        h5save('%s%s.hdf5' % (args.output_prefix,
                              _output_specs['commonspace']['output_suffix']),
               hyper.commonspace,
               compression=args.hdf5_compression)
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
                   pm, compression=args.hdf5_compression)
    if args.transform:
        tdss, dss = _transform_dss(args.transform, promappers, args)
        del dss
        verbose(1, "Store transformed datasets")
        for i, td in enumerate(tdss):
            if __debug__:
                debug('CMDLINE', "store transformed data %i: %s" % (i, str(td)))
            h5save('%s%s.hdf5' % (args.output_prefix, '_transformed%.3i' % i),
                   td, compression=args.hdf5_compression)
