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
import re
import sys

from mvpa2.base import verbose
if __debug__:
    from mvpa2.base import debug
from mvpa2.base.types import is_datasetlike

class HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == '--help':
            # lets use the manpage on mature systems ...
            try:
                import subprocess
                subprocess.check_call(
                        'man %s 2> /dev/null' % parser.prog.replace(' ', '-'),
                        shell=True)
                sys.exit(0)
            except (subprocess.CalledProcessError, OSError):
                # ...but silently fall back if it doesn't work
                pass
        if option_string == '-h':
            helpstr = "%s\n%s" \
                    % (parser.format_usage(),
                       "Use '--help' to get more comprehensive information.")
        else:
            helpstr = parser.format_help()
        # better for help2man
        helpstr = re.sub(r'optional arguments:', 'options:', helpstr)
        helpstr = re.sub(r'positional arguments:\n.*\n', '', helpstr)
        # convert all heading to have the first character uppercase
        headpat = re.compile(r'^([a-z])(.*):$',  re.MULTILINE)
        helpstr = re.subn(headpat,
               lambda match: r'{0}{1}:'.format(match.group(1).upper(),
                                             match.group(2)),
               helpstr)[0]
        # usage is on the same line
        helpstr = re.sub(r'^usage:', 'Usage:', helpstr)
        if option_string == '--help-np':
            usagestr = re.split(r'\n\n[A-Z]+', helpstr, maxsplit=1)[0]
            usage_length = len(usagestr)
            usagestr = re.subn(r'\s+', ' ', usagestr.replace('\n', ' '))[0]
            helpstr = '%s\n%s' % (usagestr, helpstr[usage_length:])
        print helpstr
        sys.exit(0)

def parser_add_common_args(parser, pos=None, opt=None, **kwargs):
    from mvpa2.cmdline import common_args
    for i, args in enumerate((pos, opt)):
        if args is None:
            continue
        for arg in args:
            arg_tmpl = globals()[arg]
            arg_kwargs = arg_tmpl[2].copy()
            arg_kwargs.update(kwargs)
            if i:
                parser.add_argument(*arg_tmpl[i], **arg_kwargs)
            else:
                parser.add_argument(arg_tmpl[i], **arg_kwargs)

def parser_add_common_opt(parser, opt, names=None, **kwargs):
    #from mvpa2.cmdline import common_args
    opt_tmpl = globals()[opt]
    opt_kwargs = opt_tmpl[2].copy()
    opt_kwargs.update(kwargs)
    if names is None:
        parser.add_argument(*opt_tmpl[1], **opt_kwargs)
    else:
        parser.add_argument(*names, **opt_kwargs)

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
                data = data[:, mask]
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

def strip_from_docstring(doc, paragraphs=None, sections=None):
    if paragraphs is None:
        paragraphs = []
    if sections is None:
        sections = []
    out = []
    # split into paragraphs
    doc = doc.split('\n\n')
    section = ''
    for par_i, par in enumerate(doc):
        lines = par.split('\n')
        if len(lines) > 1 \
           and len(lines[0]) == len(lines[1]) \
           and lines[1] == '-' * len(lines[0]):
               section = lines[0]
        if (par_i in paragraphs) or (section in sections):
            continue
        out.append(par)
    return '\n\n'.join(out)

def param2arg(parser, klass, param, arg_names=None):
    param = klass._collections_template['params'][param]
    if arg_names is None:
        arg_names = ('--%s' % param.name.replace('_', '-'),)
    # TODO add more useful information from the parameter definition
    # into the parser
    parser.add_argument(*arg_names, help=param.__doc__,
                        default=param.default,
                        type=CmdArg2ParameterType(param))

def ca2arg(parser, klass, ca, arg_names=None, help=None):
    ca = klass._collections_template['ca'][ca]
    if arg_names is None:
        arg_names = ('--%s' % ca.name.replace('_', '-'),)
    help_ = ca.__doc__
    if help:
        help_ = help_ + help
    parser.add_argument(*arg_names, help=help_, default=False,
                        action='store_true')


class CmdArg2ParameterType(object):
    def __init__(self, param):
        supported = ('allowedtype', 'min', 'max', 'name')
        for p in supported:
            varname = '_%s' % p
            if hasattr(param, p):
                setattr(self, varname, getattr(param, p))

    def __call__(self, arg):
        # interpret out type specs for parameters and return callables that convert
        # cmdline args (i.e. strings) into the required format
        # look here for more info: http://docs.python.org/library/argparse.html#type
        converted = self._arg2types(arg)
        self._range_check(converted)
        return converted

    def _range_check(self, val):
        if hasattr(self, '_min'):
            if val < self._min:
                raise argparse.ArgumentTypeError(
                        "range error "
                        "(value '%s' less than configured minimum '%s')"
                        % (val, self._min))
        if hasattr(self, '_max'):
            if val > self._max:
                raise argparse.ArgumentTypeError(
                        "range error "
                        "(value '%s' larger than configured maximum '%s')"
                        % (val, self._max))

    def _arg2types(self, arg):
        allowed = self._allowedtype
        if __debug__:
            debug('CMDLINE',
                    "type conversion for parameter '%s': %s -> %s"
                    % (self._name, arg, allowed))
        # loop over alternatives
        types = allowed.split(' or ')
        # strip out types we would not need to convert anyway
        not_convert_types = ('basestring', 'str')
        effective_types = [t for t in types if not t in not_convert_types]
        for type_ in effective_types:
            try:
                return self._arg2type(arg, type_)
            except argparse.ArgumentTypeError:
                if __debug__:
                    debug('CMDLINE', "type conversion into '%s' failed" % type_)
        # we only get here if conversion failed
        if len([t for t in types if t in not_convert_types]):
            # continuing without exception is possible
            return arg
        raise argparse.ArgumentTypeError(
                "cannot convert '%s' into '%s'" % (arg, allowed))

    def _arg2type(self, arg, allowed):
        if allowed == None:
            # we know nothing
            if __debug__:
                debug('CMDLINE',
                      "skipping parameter value conversion -- no type "
                      "spec for '%s' available" % self._name)
            return arg
        elif allowed == 'bool':
            return arg2bool(arg)
        elif allowed == 'int':
            return int(arg)
        elif allowed in ('None', 'none'):
            return arg
        elif allowed in ('float', 'float32', 'float64'):
            return float(arg)
        else:
            raise argparse.ArgumentTypeError(
                "unsupported parameter type specification: '%s'" % allowed)


def arg2bool(arg):
    arg = arg.lower()
    if arg in ['0', 'no', 'off', 'disable', 'false']:
        return False
    elif arg in ['1', 'yes', 'on', 'enable', 'true']:
        return True
    else:
        raise argparse.ArgumentTypeError(
                "'%s' cannot be converted into a boolean" % arg)

def arg2none(arg):
    arg = arg.lower()
    if arg == 'none':
        return None
    else:
        raise argparse.ArgumentTypeError(
                "'%s' cannot be converted into `None`" % arg)

def arg2learner(arg, index=0):
    from mvpa2.clfs.warehouse import clfswh
    import os.path
    if arg in clfswh.descriptions:
        # arg is a description
        return clfswh.get_by_descr(arg)
    elif os.path.isfile(arg) and arg.endswith('.py'):
        # arg is a script filepath
        return script2obj(arg)
    else:
        # warehouse tag collection?
        try:
            learner = clfswh.__getitem__(*arg.split(':'))
            if not len(learner):
                raise argparse.ArgumentTypeError(
                    "not match for given learner capabilities %s in the warehouse" % arg)
            return learner[index]
        except ValueError:
            # unknown tag
            raise argparse.ArgumentTypeError(
                "'%s' is neither a known classifier description, nor a script, "
                "nor a sequence of valid learner capabilities" % arg)

def script2obj(filepath):
    locals = {}
    execfile(filepath, dict(), locals)
    if not len(locals):
        raise argparse.ArgumentTypeError(
            "executing script '%s' did not create at least one object" % filepath)
    elif len(locals) > 1 and not 'obj' in locals:
        raise argparse.ArgumentTypeError(
            "executing script '%s' " % filepath
            + "did create multiple objects %s " % locals.keys()
            + "but none is named 'obj'")
    if len(locals) == 1:
        return locals.values()[0]
    else:
        return locals['obj']

def arg2partitioner(arg):
    arg = arg.lower()
    import mvpa2.generators.partition as part
    if arg == 'oddeven':
        return part.OddEvenPartitioner()
    elif arg == 'half':
        return part.HalfPartitioner()
    elif arg.startswith('group-'):
        ngroups = int(arg[6:])
        return part.NGroupPartitioner(ngroups)
    elif arg.startswith('n-'):
        nfolds = int(arg[2:])
        return part.NFoldPartitioner(nfolds)
    elif os.path.isfile(arg) and arg.endswith('.py'):
        # arg is a script filepath
        return script2obj(arg)
    else:
        raise argparse.ArgumentTypeError(
            "'%s' does not describe a supported partitioner type" % arg)

def arg2hdf5compression(arg):
    try:
        return int(arg)
    except:
        return arg

def ds2hdf5(ds, fname, compression=None):
    """Save one or more datasets into an HDF5 file.

    Parameters
    ----------
    ds : Datasset or list(Dataset)
      One or more datasets to store
    fname : str
      Filename of the output file. if it doesn't end with '.hdf5', such an
      extension will be appended.
    compression : {'gzip','lzf','szip'} or 1-9
      compression type for HDF5 storage. Available values depend on the specific
      HDF5 installation.
    """
    # this one doesn't actually check what it stores
    from mvpa2.base.hdf5 import h5save
    if not fname.endswith('.hdf5'):
        fname = '%s.hdf5' % fname
    verbose(1, "Save dataset to '%s'" % fname)
    h5save(fname, ds, mkdir=True, compression=compression)


def hdf2ds(fnames):
    """Load dataset(s) from an HDF5 file

    Parameters
    ----------
    fname : list(str)
      Names of the input HDF5 files

    Returns
    -------
    list(Dataset)
      All datasets-like elements in all given HDF5 files (in order of
      appearance). If any given HDF5 file contains non-Dataset elements
      they are silently ignored. If no given HDF5 file contains any
      dataset, an empty list is returned.
    """
    from mvpa2.base.hdf5 import h5load
    dss = []
    for fname in fnames:
        content = h5load(fname)
        if is_datasetlike(content):
            dss.append(content)
        else:
            for c in content:
                if is_datasetlike(c):
                    dss.append(c)
    return dss

def parser_add_common_attr_opts(parser):
    """Set up common parser options for adding dataset attributes"""
    for args in (attr_from_cmdline, attr_from_txt, attr_from_npy):
        parser_add_optgroup_from_def(parser, args)

def parser_add_optgroup_from_def(parser, defn, exclusive=False):
    """Add an entire option group from a definition in a custom format

    Returns
    -------
    parser argument group
    """
    optgrp = parser.add_argument_group(defn[0])
    if exclusive:
        rgrp = optgrp.add_mutually_exclusive_group()
    else:
        rgrp = optgrp
    for opt in defn[1]:
        rgrp.add_argument(*opt[0], **opt[1])
    return optgrp

def process_common_attr_opts(ds, args):
    """Goes through an argument namespace and processes attribute options"""
    # legacy support
    if not args.add_sa_attr is None:
        from mvpa2.misc.io.base import SampleAttributes
        smpl_attrs = SampleAttributes(args.add_sa_attr)
        for a in ('targets', 'chunks'):
            verbose(2, "Add sample attribute '%s' from sample attributes file"
                       % a)
            ds.sa[a] = getattr(smpl_attrs, a)
    if not args.add_fsl_mcpar is None:
        from mvpa2.misc.fsl.base import McFlirtParams
        mc_par = McFlirtParams(args.add_fsl_mcpar)
        for param in mc_par:
            verbose(2, "Add motion regressor as sample attribute '%s'"
                       % ('mc_' + param))
            ds.sa['mc_' + param] = mc_par[param]
    # loop over all attribute configurations that we know
    attr_cfgs = (# var, dst_collection, loader
            ('--add-sa', args.add_sa, ds.sa, _load_from_cmdline),
            ('--add-fa', args.add_fa, ds.fa, _load_from_cmdline),
            ('--add-sa-txt', args.add_sa_txt, ds.sa, _load_from_txt),
            ('--add-fa-txt', args.add_fa_txt, ds.fa, _load_from_txt),
            ('--add-sa-npy', args.add_sa_npy, ds.sa, _load_from_npy),
            ('--add-fa-npy', args.add_fa_npy, ds.fa, _load_from_npy),
        )
    for varid, srcvar, dst_collection, loader in attr_cfgs:
        if not srcvar is None:
            for spec in srcvar:
                attr_name = spec[0]
                if not len(spec) > 1:
                    raise argparse.ArgumentTypeError(
                        "%s option need at least two values " % varid +
                        "(attribute name and source filename (got: %s)" % spec)
                if dst_collection is ds.sa:
                    verbose(2, "Add sample attribute '%s' from '%s'"
                               % (attr_name, spec[1]))
                else:
                    verbose(2, "Add feature attribute '%s' from '%s'"
                               % (attr_name, spec[1]))
                attr = loader(spec[1:])
                try:
                    dst_collection[attr_name] = attr
                except ValueError, e:
                    # try making the exception more readable
                    e_str = str(e)
                    if e_str.startswith('Collectable'):
                        raise ValueError('attribute %s' % e_str[12:])
                    else:
                        raise e
    return ds

def _load_from_txt(args):
    defaults = dict(dtype=None, delimiter=None, skiprows=0, comments=None)
    if len(args) > 1:
        defaults['delimiter'] = args[1]
    if len(args) > 2:
        defaults['dtype'] = args[2]
    if len(args) > 3:
        defaults['skiprows'] = int(args[3])
    if len(args) > 4:
        defaults['comments'] = args[4]
    data = np.loadtxt(args[0], **defaults)
    return data

def _load_from_cmdline(args):
    defaults = dict(dtype='str', sep=',')
    if len(args) > 1:
        defaults['dtype'] = args[1]
    if defaults['dtype'] == 'str':
        data = [s.strip() for s in args[0].split(defaults['sep'])]
    else:
        data = np.fromstring(args[0], **defaults)
    return data

def _load_from_npy(args):
    defaults = dict(mmap_mode=None)
    if len(args) > 1 and arg2bool(args[1]):
        defaults['mmap_mode'] = 'r'
    data = np.load(args[0], **defaults)
    return data

########################
#
# common arguments
#
########################
# argument spec template
#<name> = (
#    <id_as_positional>, <id_as_option>
#    {<ArgusmentParser.add_arguments_kwargs>}
#)

help = (
    'help', ('-h', '--help', '--help-np'),
    dict(nargs=0, action=HelpAction,
         help="""show this help message and exit. --help-np forcefully disables
                 the use of a pager for displaying the help.""")
)

version = (
    'version', ('--version',),
    dict(action='version',
         help="show program's version and license information and exit")
)

multidata = (
    'data', ('-d', '--data'),
    {'nargs': '+',
     'help': 'awesome description is pending'
    }
)

data = (
    'data', ('-d', '--data'),
    {'help': 'awesome description is pending'
    }
)


multimask = (
    'masks', ('-m', '--masks'),
    {'nargs': '+'}
)

mask = (
    'mask', ('-m', '--mask'),
    {'help': 'single mask item'}
)

output_file = (
    'output', ('-o', '--output'),
    dict(type=str,
         help="""output filename ('.hdf5' extension is added automatically
        if necessary).""")
)

output_prefix = (
    'outprefix', ('-o', '--output-prefix'),
    {'type': str,
     'metavar': 'PREFIX',
     'help': 'prefix for all output file'
    }
)

classifier = (
    'classifier', ('--clf',),
    {'type': arg2learner,
     'help': """select a classifier via its description in the learner
             warehouse (see 'info' command for a listing), a colon-separated
             list of capabilities, or by a file path to a Python script that
             creates a classifier instance (advanced)."""
    }
)

partitioner = (
    'partitioner', ('--partitioner',),
    {'type': arg2partitioner,
     'help': """select a data folding scheme. Supported arguments are: 'half'
             for split-half, partitioning, 'oddeven' for partitioning into odd
             and even chunks, 'group-X' where X can be any positive integer for
             partitioning in X groups, 'n-X' where X can be any positive
             integer for leave-X-chunks out partitioning, or a file path to a
             Python script that creates a custom partitioner instance
             (advanced)."""
    }
)

hdf5compression = (
    'compression', ('--compression',),
    dict(type=arg2hdf5compression, default=None, help="""\
compression type for HDF5 storage. Available values depend on the specific HDF5
installation. Typical values are: 'gzip', 'lzf', 'szip', or integers from 1 to
9 indicating gzip compression levels."""))


attr_from_cmdline = ('options for input from the command line', [
    (('--add-sa',), dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""compose a sample attribute from the command line input.
                The first value is the desired attribute name, the second value
                is a comma-separated list (appropriately quoted) of actual
                attribute values. An optional third value can be given to
                specify a data type.
                Additional information on defining dataset attributes on the
                command line are given in the section "Compose attributes
                on the command line.""")),
    (('--add-fa',), dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""compose a feature attribute from the command line input.
                The first value is the desired attribute name, the second value
                is a comma-separated list (appropriately quoted) of actual
                attribute values. An optional third value can be given to
                specify a data type.
                Additional information on defining dataset attributes on the
                command line are given in the section "Compose attributes
                on the command line.""")),
])

attr_from_txt = ('options for input from text files', [
    (('--add-sa-txt',), dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""load sample attribute from a text file. The first value
                is the desired attribute name, the second value is the filename
                the attribute will be loaded from. Additional values modifying
                the way the data is loaded are described in the section
                "Load data from text files".""")),
    (('--add-fa-txt',), dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""load feature attribute from a text file. The first value
                is the desired attribute name, the second value is the filename
                the attribute will be loaded from. Additional values modifying
                the way the data is loaded are described in the section
                "Load data from text files".""")),
    (('--add-sa-attr',), dict(type=str, metavar='FILENAME',
        help="""load sample attribute values from an legacy 'attributes file'.
                Column data is read as "literal". Only two column files
                ('targets' + 'chunks') without headers are supported. This
                option allows for reading attributes files from early PyMVPA
                versions.""")),
])

attr_from_npy = ('options for input from stored Numpy arrays', [
    (('--add-sa-npy',), dict(type=str, nargs='+', metavar='VALUE', action='append',
        help="""load sample attribute from a Numpy .npy file. Compressed files
             (i.e. .npy.gz) are supported as well. The first value is the
             desired attribute name, the second value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
    (('--add-fa-npy',), dict(type=str, nargs='+', metavar='VALUE', action='append',
        help="""load feature attribute from a Numpy .npy file. Compressed files
             (i.e. .npy.gz) are supported as well. The first value is the
             desired attribute name, the second value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
])

single_required_hdf5output = ('output options', [
    (('-o', '--output'), dict(type=str, required=True,
         help="""output filename ('.hdf5' extension is added automatically if
         necessary).""")),
    hdf5compression[1:],
])
