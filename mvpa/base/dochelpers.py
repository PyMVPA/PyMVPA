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
        # TODO: remove common empty prefix, so we don't offset
        # documentation too much
        # to see starts/ends of the lines use
        #  return '\n'.join(['>%s<' % x for x in text.split('\n')])
        # function posixpath.commonprefix might be used to detect
        # common prefix, or just textwrap.dedent
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


def enhancedClassDocString(cls, *args):
    """Generate enhanced doc strings but given a class, not just a name.

    It is to be used from a collector, it whenever class is already created
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
                  [' '*nspaces + x for x in cls._paramsdoc.split('\n')]) + \
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

    return '\n\n'.join(docs)
