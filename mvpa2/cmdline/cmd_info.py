# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Query various information about a PyMVPA installation.

If no option is given, a  useful subset of the available information is printed.
"""

# magic line for manpage summary
# man: -*- % query various information about a PyMVPA installation

import mvpa2

__docformat__ = 'restructuredtext'

def setup_parser(parser):
    excl = parser.add_mutually_exclusive_group()
    excl.add_argument('--externals', action='store_true',
                        help='list status of external dependencies')
    if __debug__:
        excl.add_argument('--debug', action='store_true',
                          help='list available debug channels')
    excl.add_argument(
            '--learner-warehouse', nargs='*', default=False, metavar='TAG',
            help="""list available algorithms in the learner warehouse.
            Optionally, an arbitrary number of tags can be specified to
            constrain the listing to learners with matching tags.""")
    return parser

def run(args):
    if args.externals:
        print mvpa2.wtf(include=['externals'])
    elif args.debug:
        mvpa2.debug.print_registered()
    elif not args.learner_warehouse is False:
        from mvpa2.clfs.warehouse import clfswh
        clfswh.print_registered(*args.learner_warehouse)
    else:
        print mvpa2.wtf()
