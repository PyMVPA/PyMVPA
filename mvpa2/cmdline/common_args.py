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

multidata = (
    'data', ('-d', '--data'),
    {'nargs': '+',
     'help': 'awesome description is pending'
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
     'help': """select a partitioner. Supported arguments are: 'half' for
             split-half, partitioning, 'oddeven' for partitioning into odd and
             even chunks, 'group-X' where X can be any positive integer for
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
