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
from mvpa2.base.types import is_datasetlike
from mvpa2.cmdline import common_args

def parser_add_common_args(parser, pos=None, opt=None):
    if not pos is None:
        for arg in pos:
            arg_tmpl = getattr(common_args, arg)
            parser.add_argument(arg_tmpl[0], **arg_tmpl[2])
    if not opt is None:
        for arg in opt:
            arg_tmpl = getattr(common_args, arg)
            parser.add_argument(arg_tmpl[1], **arg_tmpl[2])
    return parser

def _load_if_hdf5(arg):
    # just try it, who knows whether we can trust file extensions and whether
    # we have HDF5
    try:
        from mvpa2.base.hdf5 import h5load
        return h5load(arg)
    except:
        # didn't work
        return arg

def _mask_data(data, mask):
    if is_datasetlike(data):
        if not mask is None:
            if isinstance(mask, basestring):
                raise ValueError("masks for readily loaded datasets need to be "
                                 "valid slice arguments")
            else:
                raise NotImplementedError("create static feature selection from mask")
        return data
    elif data is None:
        return None
    else:
        from mvpa2.datasets.mri import fmri_dataset
        return fmri_dataset(data, mask=mask)

def args2data(args):
    data = []
    for arg in args:
        if arg == '_none':
            data.append(None)
            continue
        res = _load_if_hdf5(arg)
        if is_datasetlike(res) or isinstance(res, basestring):
            # keep the filenames for now
            data.append(res)
        else:
            # might be a container
            data.extend(res)
    return data

def args2datasets(data_specs, mask_specs=None):
    data = args2data(data_specs)
    if mask_specs is None:
        masks = [None]
    else:
        masks = args2data(mask_specs)
        if len(masks) > 1 and len(masks) != len(data):
            raise ValueError("if more than one mask is given, their number "
                             "must equal those of data arguments")
    if len(masks) == 1:
        # duplicate for all data
        masks *= len(data)
    maskdata = zip(data, masks)
    dss = [_mask_data(*vals) for vals in maskdata]
    return dss


