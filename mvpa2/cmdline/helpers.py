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
import copy
import os

import numpy as np

from mvpa2.base import verbose
if __debug__:
    from mvpa2.base import debug
from mvpa2.base.types import is_datasetlike
from mvpa2.base.state import ClassWithCollections

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


def parser_add_common_opt(parser, opt, names=None, **kwargs):
    """Add a named option to a cmdline arg parser.

    Parameters
    ----------
    opt: str
      name of the option
    names: tuple or None
      sequence of names under which the option should be available.
      If None, the default will be used.
    """
    opt_tmpl = globals()[opt]
    opt_kwargs = opt_tmpl[2].copy()
    opt_kwargs.update(kwargs)
    if names is None:
        parser.add_argument(*opt_tmpl[1], **opt_kwargs)
    else:
        parser.add_argument(*names, **opt_kwargs)

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

def param2arg(parser, param, arg_names=None, **kwargs):
    """Convert a Node parameter into a parser argument.

    Parameters
    ----------
    parser : instance
      argparse parser instance (could be option group)
    param : tuple or parameter instance
      A length-2 tuple with the Node class (no instance) as first, and the
      name of the parameter as second element.
    arg_names : tuple(str), optional
      Argument name(s) to overwrite the default parameter name.
    **kwargs
      Any addtional options are passed on to the `add_argument()` function
      call.
    """
    if kwargs is None:
        kwargs = {}
    if isinstance(param, tuple):
        # get param instance
        param = param[0]._collections_template['params'][param[1]]
    if arg_names is None:
        # use parameter name as default argument name
        arg_names = ('--%s' % param.name.replace('_', '-'),)
    help = param.__doc__
    if param.constraints is not None:
        # allow for parameter setting overwrite via kwargs
        if not 'default' in kwargs:
            kwargs['default'] = param.default
        if not 'type' in kwargs:
            kwargs['type'] = param.constraints
        # include value contraint description and default
        # into the help string
        cdoc = param.constraints.long_description()
        if cdoc[0] == '(' and cdoc[-1] == ')':
            cdoc = cdoc[1:-1]
        help += ' Constraints: %s.' % cdoc
    try:
        help += " [Default: %r]" % (kwargs['default'],)
    except:
        pass
    # create the parameter, using the constraint instance for type
    # conversion
    parser.add_argument(*arg_names, help=help,
                        **kwargs)

def ca2arg(parser, klass, ca, arg_names=None, help=None):
    ca = klass._collections_template['ca'][ca]
    if arg_names is None:
        arg_names = ('--%s' % ca.name.replace('_', '-'),)
    help_ = ca.__doc__
    if help:
        help_ = help_ + help
    parser.add_argument(*arg_names, help=help_, default=False,
                        action='store_true')


def arg2bool(arg):
    arg = arg.lower()
    if arg in ['0', 'no', 'off', 'disable', 'false']:
        return False
    elif arg in ['1', 'yes', 'on', 'enable', 'true']:
        return True
    else:
        raise argparse.ArgumentTypeError(
                "'%s' cannot be converted into a boolean" % (arg,))

def arg2none(arg):
    arg = arg.lower()
    if arg == 'none':
        return None
    else:
        raise argparse.ArgumentTypeError(
                "'%s' cannot be converted into `None`" % (arg,))

def arg2learner(arg, index=0):
    from mvpa2.clfs.warehouse import clfswh
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
                    "not match for given learner capabilities %s in the warehouse" % (arg,))
            return learner[index]
        except ValueError:
            # unknown tag
            raise argparse.ArgumentTypeError(
                "'%s' is neither a known classifier description, nor a script, "
                "nor a sequence of valid learner capabilities" % (arg,))

def script2obj(filepath):
    locals = {}
    execfile(filepath, locals, locals)
    if not len(locals):
        raise argparse.ArgumentTypeError(
            "executing script '%s' did not create at least one object" % filepath)
    elif len(locals) > 1 and not ('obj' in locals or 'fx' in locals):
        raise argparse.ArgumentTypeError(
            "executing script '%s' " % filepath
            + "did create multiple objects %s " % locals.keys()
            + "but none is named 'obj' or 'fx'")
    if len(locals) == 1:
        return locals.values()[0]
    else:
        if 'obj' in locals:
            return locals['obj']
        else:
            return locals['fx']

def arg2partitioner(arg):
    # check for an optional 'attr' argument
    args = arg.split(':')
    arg = args[0]
    if len(args) == 1:
        chunk_attr = 'chunks'
    else:
        chunk_attr = ':'.join(args[1:])
    arglower = arg.lower()
    import mvpa2.generators.partition as part
    if arglower == 'oddeven':
        return part.OddEvenPartitioner(attr=chunk_attr)
    elif arglower == 'half':
        return part.HalfPartitioner(attr=chunk_attr)
    elif arglower.startswith('group-'):
        ngroups = int(arglower[6:])
        return part.NGroupPartitioner(ngroups, attr=chunk_attr)
    elif arglower.startswith('n-'):
        nfolds = int(arglower[2:])
        return part.NFoldPartitioner(nfolds, attr=chunk_attr)
    elif os.path.isfile(arg) and arg.endswith('.py'):
        # arg is a script filepath
        return script2obj(arg)
    else:
        raise argparse.ArgumentTypeError(
            "'%s' does not describe a supported partitioner type" % arg)

def arg2errorfx(arg):
    import mvpa2.misc.errorfx as efx
    if hasattr(efx, arg):
        return getattr(efx, arg)
    elif os.path.isfile(arg) and arg.endswith('.py'):
        # arg is a script filepath
        return script2obj(arg)
    else:
        raise argparse.ArgumentTypeError(
            "'%s' does not describe a supported error function" % arg)

def arg2hdf5compression(arg):
    try:
        return int(arg)
    except:
        return arg

def arg2neighbor(arg):
    # [[shape:]shape:]params
    comp = arg.split(':')
    if not len(comp):
        # need at least a radius
        raise ValueError("incomplete neighborhood specification")
    if len(comp) == 1:
        # [file|sphere radius]
        attr = 'voxel_indices'
        arg = comp[0]
        if os.path.isfile(arg) and arg.endswith('.py'):
            neighbor = script2obj(arg)
        else:
            from mvpa2.misc.neighborhood import Sphere
            neighbor = Sphere(int(arg))
    elif len(comp) == 2:
        # attr:[file|sphere radius]
        attr = comp[0]
        arg = comp[1]
        if os.path.isfile(arg) and arg.endswith('.py'):
            neighbor = script2obj(arg)
        else:
            from mvpa2.misc.neighborhood import Sphere
            neighbor = Sphere(int(arg))
    elif len(comp) > 2:
        attr = comp[0]
        shape = comp[1]
        params = [float(c) for c in comp[2:]]
        import mvpa2.misc.neighborhood as neighb
        neighbor = getattr(neighb, shape)(*params)
    return attr, neighbor

def ds2hdf5(ds, fname, compression=None):
    """Save one or more datasets into an HDF5 file.

    Parameters
    ----------
    ds : Dataset or list(Dataset)
      One or more datasets to store
    fname : str
      Filename of the output file. If it doesn't end with '.hdf5', such an
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

def arg2ds(sources):
    """Convert a sequence of dataset sources into a dataset.

    This function would be used to used to convert a single --input
    multidata specification into a dataset. For multiple --input
    arguments execute this function in a loop.
    """
    from mvpa2.base.dataset import vstack
    return vstack(hdf2ds(sources))

def parser_add_common_attr_opts(parser):
    """Set up common parser options for adding dataset attributes"""
    for args in (attr_from_cmdline, attr_from_txt, attr_from_npy):
        parser_add_optgroup_from_def(parser, args)

def parser_add_optgroup_from_def(parser, defn, exclusive=False, prefix=None):
    """Add an entire option group from a definition in a custom format

    Parameters
    ----------
    parser : argparser instance
    defn : tuple
      Option group spec. Complicated beast. Grep source code for syntax examples.
    exclusive : bool
      Flag to make all options in the group mutually exclusive.
    prefix : str
      Prefix all option names with this string.

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
        namespec = opt[0]
        param = None
        if len(namespec) == 2 and not isinstance(namespec[0], basestring) \
          and issubclass(namespec[0], ClassWithCollections):
            # parameter spec -> use its name
            param = namespec[0]._collections_template['params'][namespec[1]]
            optnames = ('--%s' % param.name.replace('_', '-'),)
        else:
            # take the literal names
            optnames = namespec
        if not prefix is None:
            # overwrite all option names with a common prefix
            optnames = ['%s%s' % (prefix, on.lstrip('-')) for on in optnames]
        if param is None and len(opt) > 1 and not isinstance(opt[1], dict):
            # parameter spec is given at 2nd position
            param = opt[1]
        if isinstance(opt[-1], dict):
            # last element has kwags for add_argument
            add_kwargs = opt[-1]
        else:
            # nothing to add
            add_kwargs = {}
        if not param is None:
            param2arg(rgrp, param, arg_names=optnames, **add_kwargs)
        else:
            rgrp.add_argument(*optnames, **add_kwargs)
    return optgrp

def process_common_dsattr_opts(ds, args):
    """Goes through an argument namespace and processes attribute options"""
    # legacy support
    if not args.add_sa_attr is None:
        from mvpa2.misc.io.base import SampleAttributes
        smpl_attrs = SampleAttributes(args.add_sa_attr)
        for a in ('targets', 'chunks'):
            verbose(2, "Add sample attribute '%s' from sample attributes file"
                       % a)
            ds.sa[a] = getattr(smpl_attrs, a)
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
        import numpy as np
        data = np.fromstring(args[0], **defaults)
    return data

def _load_from_npy(args):
    defaults = dict(mmap_mode=None)
    if len(args) > 1 and arg2bool(args[1]):
        defaults['mmap_mode'] = 'r'
    data = np.load(args[0], **defaults)
    return data

def _load_csv_table(f):
    import csv
    import numpy as np
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(f.read(1024))
    except:
        # maybe a sloppy header with a trailing delimiter?
        f.seek(0)
        sample = [f.readline() for s in range(3)]
        sample[0] = sample[0].strip()
        dialect = sniffer.sniff('\n'.join(sample))
    f.seek(0)
    reader = csv.DictReader(f, dialect=dialect)
    table = dict(zip(reader.fieldnames,
                       [list() for i in xrange(len(reader.fieldnames))]))
    for row in reader:
        for k, v in row.iteritems():
            table[k].append(v)
    del_me = []
    for k, v in table.iteritems():
        if not len(k) and len(v) and v[0] is None:
            # this is an artifact of a trailing delimiter
            del_me.append(k)
        try:
            table[k] = np.array(v, dtype=int)
        except ValueError:
            try:
                table[k] = np.array(v, dtype=float)
            except ValueError:
                # we tried ...
                pass
        except TypeError:
            # tolerate any unexpected types and keep them as is
            pass
    for d in del_me:
        # delete artifacts
        del table[d]
    return table

def get_crossvalidation_instance(learner, partitioner, errorfx,
                                 sampling_repetitions=1,
                                 learner_space='targets',
                                 balance_training=None,
                                 permutations=0,
                                 avg_datafold_results=True,
                                 prob_tail='left'):
    from mvpa2.base.node import ChainNode
    from mvpa2.measures.base import CrossValidation
    if not balance_training is None:
        # balance training data
        try:
            amount = int(balance_training)
        except ValueError:
            try:
                amount = float(balance_training)
            except ValueError:
                amount = balance_training
        from mvpa2.generators.resampling import Balancer
        balancer = Balancer(amount=amount, attr=learner_space,
                            count=sampling_repetitions,
                            limit={partitioner.get_space(): 1},
                            apply_selection=True,
                            include_offlimit=True)
    else:
        balancer = None
    # set learner space
    learner.set_space(learner_space)
    # setup generator for data folding -- put in a chain node for easy
    # amending
    gennode = ChainNode([partitioner], space=partitioner.get_space())
    if avg_datafold_results:
        from mvpa2.mappers.fx import mean_sample
        postproc = mean_sample()
    else:
        postproc = None
    if not balancer is None:
        # enable balancing step for each partitioning step
        gennode.append(balancer)
    if permutations > 0:
        from mvpa2.generators.base import Repeater
        from mvpa2.generators.permutation import AttributePermutator
        from mvpa2.clfs.stats import MCNullDist
        # how often do we want to shuffle the data
        repeater = Repeater(count=permutations)
        # permute the training part of a dataset exactly ONCE
        permutator = AttributePermutator(
                        learner_space,
                        limit={partitioner.get_space(): 1},
                        count=1)
        # CV with null-distribution estimation that permutes the training data for
        # each fold independently
        perm_gen_node = copy.deepcopy(gennode)
        perm_gen_node.append(permutator)
        null_cv = CrossValidation(learner,
                                  perm_gen_node,
                                  postproc=postproc,
                                  errorfx=errorfx)
        # Monte Carlo distribution estimator
        distr_est = MCNullDist(repeater,
                               tail=prob_tail,
                               measure=null_cv,
                               enable_ca=['dist_samples'])
        # pass the p-values as feature attributes on to the results
        pass_attr = [('ca.null_prob', 'fa', 1)]
    else:
        distr_est = None
        pass_attr = None
    # final CV node
    cv = CrossValidation(learner,
                         gennode,
                         errorfx=errorfx,
                         null_dist=distr_est,
                         postproc=postproc,
                         enable_ca=['stats', 'null_prob'],
                         pass_attr=pass_attr)
    return cv



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
    'data', ('-i', '--input'),
    {'nargs': '+',
     'dest': 'data',
     'metavar': 'DATASET',
     'help': """path(s) to one or more PyMVPA dataset files. All datasets
             will be merged into a single dataset (vstack'ed) in order of
             specification. In some cases this option may need to be specified
             more than once if multiple, but separate, input datasets are
             required."""
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

learner_opt = (
    'learner', ('--learner',),
    {'type': arg2learner,
     'help': """select a learner (trainable node) via its description in the
             learner warehouse (see 'info' command for a listing), a
             colon-separated list of capabilities, or by a file path to a Python
             script that creates a classifier instance (advanced)."""
    }
)

learner_space_opt = (
    'learnerspace', ('--learner-space',),
    {'type': str, 'default': 'targets',
     'help': """name of a sample attribute that defines the model to be
             learned by a learner. By default this is an attribute named
             'targets'."""
    }
)

partitioner_opt = (
    'partitioner', ('--partitioner',),
    {'type': arg2partitioner,
     'help': """select a data folding scheme. Supported arguments are: 'half'
             for split-half partitioning, 'oddeven' for partitioning into odd
             and even chunks, 'group-X' where X can be any positive integer for
             partitioning in X groups, 'n-X' where X can be any positive
             integer for leave-X-chunks out partitioning. By default
             partitioners operate on dataset chunks that are defined by a
             'chunks' sample attribute. The name of the "chunking" attribute
             can be changed by appending a colon and the name of the attribute
             (e.g. 'oddeven:run'). optionally an argument to this option can
             also be a file path to a Python script that creates a custom
             partitioner instance (advanced)."""
    }
)

enable_ca_opt = (
    'enable_ca', ('--enable-ca',),
    {'nargs': '+', 'metavar': 'NAME',
     'help': """list of conditional attributes to be enabled"""
    }
)

disable_ca_opt = (
    'disable_ca', ('--disable-ca',),
    {'nargs': '+', 'metavar': 'NAME',
     'help': """list of conditional attributes to be disabled"""
    }
)

ca_opts_grp = ('options for conditional attributes',
        [enable_ca_opt[1:], disable_ca_opt[1:]])

hdf5compression = (
    'compression', ('--hdf5-compression',),
    dict(type=arg2hdf5compression, default=None, metavar='TYPE', help="""\
compression type for HDF5 storage. Available values depend on the specific HDF5
installation. Typical values are: 'gzip', 'lzf', 'szip', or integers from 1 to
9 indicating gzip compression levels."""))


attr_from_cmdline = ('options for attributes from the command line', [
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

attr_from_txt = ('options for attributes from text files', [
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

attr_from_npy = ('options for attributes from stored Numpy arrays', [
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
         necessary). NOTE: The output format is suitable for data exchange between
         PyMVPA commands, but is not recommended for long-term storage or exchange
         as its specific content may vary depending on the actual software
         environment. For long-term storage consider conversion into other data
         formats (see 'dump' command).""")),
    hdf5compression[1:],
])

crossvalidation_opts_grp = ('options for cross-validation setup', [
    learner_opt[1:], learner_space_opt[1:], partitioner_opt[1:],
    (('--errorfx',), dict(type=arg2errorfx,
        help="""error function to be applied to the targets and predictions
        of each cross-validation data fold. This can either be a name of
        any error function in PyMVPA's mvpa2.misc.errorfx module, or a file
        path to a Python script that creates a custom error function
        (advanced).""")),
    (('--avg-datafold-results',), dict(action='store_true',
        help="""average result values across data folds generated by the
        partitioner. For example to compute a mean prediction error across
        all folds of a cross-validation procedure.""")),
    (('--balance-training',), dict(type=str,
        help="""If enabled, training samples are balanced within each data fold.
        If the keyword 'equal' is given as argument an equal number of random
        samples for each unique target value is chosen. The number of samples
        per category is determined by the category with the least number of
        samples in the respective training set. An integer argument will cause
        the a corresponding number of samples per category to be randomly
        selected. A floating point number argument (interval [0,1]) indicates
        what fraction of the available samples shall be selected.""")),
    (('--sampling-repetitions',), dict(type=int, default=1,
        help="""If training set balancing is enabled, how often should random
        sample selection be performed for each data fold. Default: 1""")),
    (('--permutations',), dict(type=int, default=0,
        help="""Number of Monte-Carlo permutation runs to be computed for
        estimating an H0 distribution for all cross-validation results. Enabling
        this option will make reports of corresponding p-values available in
        the result summary and output.""")),
    (('--prob-tail',), dict(choices=('left', 'right'), default='left',
        help="""which tail of the probability distribution to report p-values
        from when evaluating permutation test results. For example, a
        cross-validation computing mean prediction error could report left-tail
        p-value for a single-sided test.""")),
])
