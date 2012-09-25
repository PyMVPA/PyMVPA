# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

__docformat__ = 'restructuredtext'

import argparse
from mvpa2.base import verbose
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
        import parser_add_common_args, args2datasets, strip_from_docstring, \
               param2arg, ca2arg

from mvpa2.algorithms.hyperalignment import Hyperalignment

parser_args = {
    'description': strip_from_docstring(Hyperalignment.__doc__,
                                        paragraphs=(4,),
                                        sections=(('Examples', 'Notes'))),
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

_supported_cas = (
    'residual_errors', 'training_residual_errors',
)
_supported_parameters = (
    'alpha', 'level2_niter', 'ref_ds', 'zscore_all', 'zscore_common',
)

def setup_parser(parser):
    # order of calls is relevant!
    inputargs = parser.add_argument_group('input data arguments')
    parser_add_common_args(inputargs, pos=['multidata'], opt=['multimask'])
    algoparms = parser.add_argument_group('algorithm parameters')
    for param in _supported_parameters:
        param2arg(algoparms, Hyperalignment, param)
    outputopts = parser.add_argument_group('output options')
    for ca in _supported_cas:
        ca2arg(outputopts, Hyperalignment, ca)


def run(args):
    if __debug__:
        debug('CMDLINE', "loading input data from %s" % args.data)
    dss = args2datasets(args.data, args.masks)
    verbose(1, "Loaded %i input datasets" % len(dss))
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
    # think about how to output the results
    #print [str(ds) for ds in hyper.ca.residual_errors]



