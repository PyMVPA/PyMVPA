# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Execute arbitrary Python expressions (on datasets)

This command can be used to execute arbitrary Python scripts while avoiding
unnecessary boilerplate code to load datasets and store results. This command
is also useful for testing functionality and results via the commandline
interface and for asserting arbitrary conditions in scripts.

First, optional dataset(s) are loaded from one or more sources. Afterwards any
number of given expressions (see --exec) are executed. An expression can be
given as an argument on the command line, read from a file, or from STDIN. The
return value of any given expression is ignored (not evaluated anyhow), only
exceptions are treated as errors and cause the command to exit with a non-zero
return value.  To implement tests and assertions it is best to utilize a Python
unittest framework such as 'nose'.

In the namespace in which all expressions are evaluated the NumPy module is
available via the alias 'np', and the nose.tools under the alias 'nt' (if
installed). Any loaded datasets are available as a list named ``dss``. The
first dataset in that list (if any) is available under the name ``ds``.

Examples:

Assert some condition

  $ pymvpa2 exec -e 'assert(4==4)'

Check for the presence of a particular sample attribute in a dataset

  $ pymvpa2 exec -e 'dss[0].sa.subj3' -i mydata.hdf5

Extract and store results

  $ pymvpa2 exec -e 'a=5' -e 'print a' --store a -o mylittlea.hdf5
"""

# magic line for manpage summary
# man: -*- % evaluate arbitrary Python expressions for tests and assertions

__docformat__ = 'restructuredtext'

import os
import sys
import numpy as np
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import vstack
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers import arg2ds, ds2hdf5, parser_add_common_opt, \
        hdf5compression, parser_add_optgroup_from_def

hdf5output = ('output options', [
    (('-s', '--store'), dict(type=str, nargs='+', metavar='NAME', help="""\
        One or more names of variables or objects to extract from the local
        name space after all expressions have been executed. They will be
        stored in a dictionary in HDF5 format (requires --output).""")),
    (('-o', '--output'), dict(type=str,
         help="""output filename ('.hdf5' extension is added automatically if
         necessary). NOTE: The output format is suitable for data exchange between
         PyMVPA commands, but is not recommended for long-term storage or exchange
         as its specific content may vary depending on the actual software
         environment. For long-term storage consider conversion into other data
         formats (see 'dump' command).""")),
    hdf5compression[1:],
])



parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

def setup_parser(parser):
    parser_add_common_opt(parser, 'multidata', nargs='*', action='append')
    parser.add_argument('-e', '--exec', type=str, required=True,
            metavar='EXPR', action='append', dest='eval',
            help="""Python expression, or filename of a Python script,
            or '-' to read expressions from STDIN.""")
    parser_add_optgroup_from_def(parser, hdf5output)

def run(args):
    if args.store is not None and args.output is None:
        raise ValueError("--output is require for result storage")
    if args.data is not None:
        dss = [arg2ds(d) for d in args.data]
        if len(dss):
            # convenience short-cut
            ds = dss[0]
    try:
        import nose.tools as nt
    except ImportError:
        pass
    for expr in args.eval:
        if expr == '-':
            exec sys.stdin
        elif os.path.isfile(expr):
            execfile(expr, globals(), locals())
        else:
            exec expr
    if args.store is not None:
        out = {}
        for var in args.store:
            try:
                out[var] = locals()[var]
            except KeyError:
                warning("'%s' not found in local name space -- skipped." % var)
        if len(out):
            ds2hdf5(out, args.output, compression=args.hdf5_compression)


