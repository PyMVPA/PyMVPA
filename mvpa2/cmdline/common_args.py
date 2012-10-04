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

