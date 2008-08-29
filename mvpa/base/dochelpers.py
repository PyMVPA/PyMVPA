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

import re, textwrap

# for table2string
import numpy as N
from math import ceil

def rstUnderline(text, markup):
    """Add and underline RsT string matching the length of the given string.
    """
    return text + '\n' + markup * len(text)


def handleDocString(text):
    """Take care of empty and non existing doc strings."""
    if text == None or not len(text):
        return 'No documentation found. Sorry!'
    else:
        # TODO: remove common empty prefix, so we don't offset
        # documentation too much
        # to see starts/ends of the lines use
        #  return '\n'.join(['>%s<' % x for x in text.split('\n')])
        # function posixpath.commonprefix might be used to detect
        # common prefix, or just textwrap.dedent
        # Problem is that first line might often have no offset, so might
        # need to be ignored from dedent call
        if not text.startswith(' '):
            lines = text.split('\n')
            text2 = '\n'.join(lines[1:])
            return lines[0] + "\n" + textwrap.dedent(text2)
        else:
            return textwrap.dedent(text)


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


def enhancedClassDocString(cls, *args):
    """Generate enhanced doc strings but given a class, not just a name.

    It is to be used from a collector, ie whenever class is already created
    """
    name = cls.__name__
    lcl = cls.__dict__
    rst_lvlmarkup = ["=", "-", "_"]

    initdoc = None
    if lcl.has_key('__init__'):
        initdoc = lcl['__init__'].__doc__

    if lcl.has_key('_paramsdoc'):
        if initdoc is None:
            initdoc = "Initialize instance of %s" % name

        # collector provided us with documentation for the parameters
        if not (":Parameters:" in initdoc):
            initdoc += "\n\n:Parameters:\n"


        # where new line is after :Parameters:
        nl_index = initdoc.index('\n', initdoc.index(':Parameters:')+1)
        # how many spaces preceed next line
        initdoc_therest = initdoc[nl_index+1:]
        nspaces = len(initdoc_therest) - len(initdoc_therest.lstrip())
        initdoc = initdoc[:nl_index+1] + '\n'.join(
                  [' '*(nspaces-2) + x for x in cls._paramsdoc.rstrip().split('\n')]) + \
                  initdoc[nl_index:]

    docs = []
    docs += [ handleDocString(lcl['__doc__']),
              rstUnderline('Constructor information for `%s` class' % name,
                           rst_lvlmarkup[2]),
              handleDocString(initdoc) ]

    if len(args):
        docs.append(rstUnderline('\nDocumentation for base classes of `%s`' \
                                 % name, rst_lvlmarkup[0]))
    for i in args:
        docs += [ rstUnderline('Documentation for class `%s`' % i.__name__,
                               rst_lvlmarkup[1]),
                  handleDocString(i.__doc__) ]

    result = '\n\n'.join(docs)
    # remove some bogus new lines -- never 3 empty lines in doc are useful
    result = re.sub("\s*\n\s*\n\s*\n", "\n\n", result)

    return result


def table2string(table, out=None):
    """Given list of lists figure out their common widths and print to out

    :Parameters:
      table : list of lists of strings
        What is aimed to be printed
      out : None or stream
        Where to print. If None -- will print and return string

    :Returns:
      string if out was None
    """

    print2string = out is None
    if print2string:
        out = StringIO()

    # equalize number of elements in each row
    Nelements_max = max(len(x) for x in table)
    for i,table_ in enumerate(table):
        table[i] += [''] * (Nelements_max - len(table_))

    # figure out lengths within each column
    atable = N.asarray(table)
    col_width = [ max( [len(x) for x in column] ) for column in atable.T ]
    string = ""
    for i, table_ in enumerate(table):
        string_ = ""
        for j, item in enumerate(table_):
            item = str(item)
            NspacesL = ceil((col_width[j] - len(item))/2.0)
            NspacesR = col_width[j] - NspacesL - len(item)
            string_ += "%%%ds%%s%%%ds " \
                       % (NspacesL, NspacesR) % ('', item, '')
        string += string_.rstrip() + '\n'
    out.write(string)

    if print2string:
        value = out.getvalue()
        out.close()
        return value

    pass

