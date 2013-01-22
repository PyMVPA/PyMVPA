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

# argument spec template
#<name> = (
#    <id_as_positional>, <id_as_option>
#    {<ArgusmentParser.add_arguments_kwargs>}
#)

from mvpa2.cmdline.helpers import HelpAction
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

from mvpa2.cmdline.helpers import arg2learner
classifier = (
    'classifier', ('--clf',),
    {'type': arg2learner,
     'help': """select a classifier via its description in the learner
             warehouse (see 'info' command for a listing), a colon-separated
             list of capabilities, or by a file path to a Python script that
             creates a classifier instance (advanced)."""
    }
)

from mvpa2.cmdline.helpers import arg2partitioner
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

from mvpa2.cmdline.helpers import arg2hdf5compression
hdf5compression = (
    'compression', ('--compression',),
    dict(type=arg2hdf5compression, default=None, help="""\
compression type for HDF5 storage. Available values depend on the specific HDF5
installation. Typical values are: 'gzip', 'lzf', 'szip', or integers from 1 to
9 indicating gzip compression levels."""))


attr_from_cmdline = ('options for input from the command line', [
    ('--add-sa', dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""compose a sample attribute from the command line input.
                The first value is the desired attribute name, the second value
                is a comma-separated list (appropriately quoted) of actual
                attribute values. An optional third value can be given to
                specify a data type.
                Additional information on defining dataset attributes on the
                command line are given in the section "Compose attributes
                on the command line.""")),
    ('--add-fa', dict(type=str, nargs='+', action='append', metavar='VALUE',
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
    ('--add-sa-txt', dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""load sample attribute from a text file. The first value
                is the desired attribute name, the second value is the filename
                the attribute will be loaded from. Additional values modifying
                the way the data is loaded are described in the section
                "Load data from text files".""")),
    ('--add-fa-txt', dict(type=str, nargs='+', action='append', metavar='VALUE',
        help="""load feature attribute from a text file. The first value
                is the desired attribute name, the second value is the filename
                the attribute will be loaded from. Additional values modifying
                the way the data is loaded are described in the section
                "Load data from text files".""")),
    ('--add-sa-attr', dict(type=str, metavar='FILENAME',
        help="""load sample attribute values from an legacy 'attributes file'.
                Column data is read as "literal". Only two column files
                ('targets' + 'chunks') without headers are supported. This
                option allows for reading attributes files from early PyMVPA
                versions.""")),
])

attr_from_npy = ('options for input from stored Numpy arrays', [
    ('--add-sa-npy', dict(type=str, nargs='+', metavar='VALUE', action='append',
        help="""load sample attribute from a Numpy .npy file. Compressed files
             (i.e. .npy.gz) are supported as well. The first value is the
             desired attribute name, the second value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
    ('--add-fa-npy', dict(type=str, nargs='+', metavar='VALUE', action='append',
        help="""load feature attribute from a Numpy .npy file. Compressed files
             (i.e. .npy.gz) are supported as well. The first value is the
             desired attribute name, the second value is the filename
             the data will be loaded from. Additional values modifying the way
             the data is loaded are described in the section "Load data from
             Numpy NPY files".""")),
])


