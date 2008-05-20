#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

__docformat__ = 'restructuredtext'


def rstUnderline(text, markup):
    """Add and underline RsT string matching the length of the given string.
    """
    return text + '\n' + markup * len(text)


def handleDocString(text):
    """Take care of empty and non existing doc strings."""
    if text == None or not len(text):
        return 'No documentation found. Sorry!'
    else:
        return text


def enhancedDocString(name, lcl, *args):
    """Generate enhanced doc strings."""
    rst_lvlmarkup = ["=", "-", "_"]

    docs = []
    docs += [ handleDocString(lcl['__doc__']),
              rstUnderline('Constructor information for `%s` class' % name,
                           rst_lvlmarkup[2]),
              handleDocString(lcl['__init__'].__doc__) ]

    if len(args):
        docs.append(rstUnderline('\nDocumentation for base classes of `%s`' \
                                 % name, rst_lvlmarkup[0]))
    for i in args:
        docs += [ rstUnderline('Documentation for class `%s`' % i.__name__,
                               rst_lvlmarkup[1]),
                  handleDocString(i.__doc__) ]

    return '\n\n'.join(docs)
