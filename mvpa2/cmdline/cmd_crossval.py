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
        import parser_add_common_args, parser_add_common_opt
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.measures.base import CrossValidation
from mvpa2.base.node import ChainNode

def setup_parser(parser):
    # order of calls is relevant!
    parser_add_common_args(parser, pos=['data'])
    parser_add_common_opt(parser, 'classifier',
                          names=('-c', '--clf', '--classifier'),
                          required=True)
    parser_add_common_opt(parser, 'partitioner',
                          names=('-f', '--fold'),
                          required=True)

def run(args):
    ds = h5load(args.data)
    # XXX remove me
    ds = ds[:,:2]
    gennode = ChainNode([args.fold], space=args.fold.get_space())
    cv = CrossValidation(args.clf, gennode, enable_ca=['stats'])
    res = cv(ds)
    raise NotImplementedError
    print cv.ca.stats
    print res
    print res.samples
    return res
