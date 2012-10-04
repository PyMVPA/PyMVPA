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

from mvpa2.base import verbose
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
        import parser_add_common_args


def setup_parser(parser):
    # order of calls is relevant!
    parser_add_common_args(parser, pos=['multidata'],
                           opt=['multimask', 'classifier'])

def run(args):
    print args.clf
