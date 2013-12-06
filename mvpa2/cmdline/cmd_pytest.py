# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Evaluate arbitrary Python expressions for tests and assertions

This command is mostly useful for testing functionality and results via the
commandline interface and for asserting arbitrary conditions in scripts.

First, a optional dataset is loaded from one or more source. Afterwards a given
expression (see --eval) is evaluated. Such an expression can be given as an
argument on the command line, read from a file, or from STDIN. The return value
of any given expression is ignored (not evaluated anyhow), only exceptitons are
treated as errors and cause the command to exit with a non-zero return value.
To implement tests and assertions is is best to utilize a Python unittest
framework such as 'nose'.

In the namespace in which all expressions are evaluated the NumPy module is
available via the alias 'np', and the nose.tools under the alias 'nt' (if
installed).

Examples:

Assert some condition

  $ pymvpa2 pytest -e 'assert(4==4)'

Check for the presence of a particular sample attribute in a dataset

  $ pymvpa2 pytest -e 'ds.sa.subj3' mydata.hdf5

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
from mvpa2.cmdline.helpers import hdf2ds, parser_add_common_opt


parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

def setup_parser(parser):
    parser_add_common_opt(parser, 'multidata', nargs='*')
    parser.add_argument('-e', '--eval', type=str, required=True,
            metavar='EXPR',
            help="""Python expression, or filename of a Python script,
            or '-' to read expressions from STDIN.""")

def run(args):
    if not args.data is None:
        dss = hdf2ds(args.data)
        ds = vstack(dss)
    try:
        import nose.tools as nt
    except ImportError:
        pass
    if args.eval == '-':
        exec sys.stdin
    elif os.path.isfile(args.eval):
        execfile(args.eval, globals())
    else:
        exec args.eval

