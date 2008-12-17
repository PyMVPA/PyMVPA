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
from StringIO import StringIO

from mvpa.base import externals
if __debug__:
    from mvpa.base import debug

__add_init2doc = False
__in_ipython = externals.exists('in ipython')
# if ran within IPython -- might need to add doc to init
if __in_ipython:
    from IPython import Release
    # XXX figure out exact version when init doc started to be added to class
    # description
    if Release.version <= '0.8.1':
        __add_init2doc = True

def rstUnderline(text, markup):
    """Add and underline RsT string matching the length of the given string.
    """
    return text + '\n' + markup * len(text)


def handleDocString(text, polite=True):
    """Take care of empty and non existing doc strings."""
    if text == None or not len(text):
        if polite:
            return 'No documentation found. Sorry!'
        else:
            return ''
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


def _indent(text, istr='  '):
    """Simple indenter
    """
    return '\n'.join(istr + s for s in text.split('\n'))


def __enhancedDocString_deprecated(name, lcl, *args):
    """Generate enhanced doc strings."""
    return lcl['__doc__']
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


def _splitOutParametersStr(initdoc):
    """ header, parameters, suffix <- initdoc
    """
    if not (":Parameters:" in initdoc):
        result = initdoc, "", ""
    else:
        # XXX any why not just re ???
        # where new line is after :Parameters:
        # parameters header index
        ph_i = initdoc.index(':Parameters:')

        # parameters body index
        pb_i = initdoc.index('\n', ph_i+1)

        # end of parameters
        try:
            pe_i = initdoc.index('\n\n', pb_i)
        except ValueError:
            pe_i = len(initdoc)

        result = initdoc[:ph_i].rstrip('\n '), initdoc[pb_i:pe_i], initdoc[pe_i:]

    # XXX a bit of duplication of effort since handleDocString might
    # do splitting internally
    return [handleDocString(x, polite=False).strip('\n') for x in result]


__re_params = re.compile('(?:\n\S.*?)+$')
__re_spliter1 = re.compile('(?:\n|\A)(?=\S)')
__re_spliter2 = re.compile('[\n:]')
def _parseParameters(paramdoc):
    """Parse parameters and return list of (name, full_doc_string)

    It is needed to remove multiple entries for the same parameter
    like it could be with adding parameters from the parent class

    It assumes that previousely parameters were unwrapped, so their
    documentation starts at the begining of the string, like what
    should it be after _splitOutParametersStr
    """
    entries = __re_spliter1.split(paramdoc)
    result = [(__re_spliter2.split(e)[0].strip(), e) for e in entries if e != '']
    if __debug__:
        debug('DOCH', 'parseParameters: Given "%s", we split into %s' %
              (paramdoc, result))
    return result


def enhancedDocString(item, *args, **kwargs):
    """Generate enhanced doc strings for various items.

    :Parameters:
      item : basestring or class
        What object requires enhancing of documentation
      *args : list
        Includes base classes to look for parameters, as well, first item
        must be a dictionary of locals if item is given by a string
      force_extend : bool
        Either to force looking for the documentation in the parents.
        By default force_extend = False, and lookup happens only if kwargs
        is one of the arguments to the respective function (e.g. item.__init__)
      skip_params : list of basestring
        List of parameters (in addition to [kwargs]) which should not
        be added to the documentation of the class.

    It is to be used from a collector, ie whenever class is already created
    """
    # Handling of arguments
    if len(kwargs):
        if set(kwargs.keys()).issubset(set(['force_extend'])):
            raise ValueError, "Got unknown keyword arguments (smth among %s)" \
                  " in enhancedDocString." % kwargs
    force_extend = kwargs.get('force_extend', False)
    skip_params = kwargs.get('skip_params', [])

    # XXX make it work also not only with classes but with methods as well
    if isinstance(item, basestring):
        if len(args)<1 or not isinstance(args[0], dict):
            raise ValueError, \
                  "Please provide locals for enhancedDocString of %s" % item
        name = item
        lcl = args[0]
        args = args[1:]
    elif hasattr(item, "im_class"):
        # bound method
        raise NotImplementedError, "enhancedDocString is not yet implemented for methods"
    elif hasattr(item, "__name__"):
        name = item.__name__
        lcl = item.__dict__
    else:
        raise ValueError, "Don't know how to extend docstring for %s" % item

    #return lcl['__doc__']
    rst_lvlmarkup = ["=", "-", "_"]

    initdoc = ""
    if lcl.has_key('__init__'):
        func = lcl['__init__']
        initdoc = func.__doc__

        # either to extend arguments
        # do only if kwargs is one of the arguments
        extend_args = force_extend or 'kwargs' in func.func_code.co_names

        if __debug__ and not extend_args:
            debug('DOCH', 'Not extending parameters for %s' % name)

        if initdoc is None:
            initdoc = "Initialize instance of %s" % name

        initdoc, params, suffix = _splitOutParametersStr(initdoc)

        if lcl.has_key('_paramsdoc'):
            params += '\n' + handleDocString(lcl['_paramsdoc'])

        params_list = _parseParameters(params)
        known_params = set([i[0] for i in params_list])
        # no need for placeholders
        skip_params = set(skip_params + ['kwargs', '**kwargs'])

        # XXX we do evil check here, refactor code to separate
        #     regressions out of the classifiers, and making
        #     retrainable flag not available for those classes which
        #     can't actually do retraining. Although it is not
        #     actually that obvious for Meta Classifiers
        if hasattr(item, '_clf_internals'):
            clf_internals = item._clf_internals
            for i in ('regression', 'retrainable'):
                if not i in item._clf_internals:
                    skip_params.update([i])

        known_params.update(skip_params)
        if extend_args:
            # go through all the parents and obtain their init parameters
            parent_params_list = []
            for i in args:
                if hasattr(i, '__init__'):
                    # XXX just assign within a class to don't redo without need
                    initdoc_ = i.__init__.__doc__
                    if initdoc_ is None:
                        continue
                    initdoc_, params_, suffix_ = _splitOutParametersStr(initdoc_)
                    parent_params_list += _parseParameters(params_.lstrip())

            # extend with ones which are not known to current init
            for i,v in parent_params_list:
                if not (i in known_params):
                    params_list += [(i,v)]
                    known_params.update([i])

        # if there are parameters -- populate the list
        if len(params_list):
            params_ = '\n'.join([i[1].rstrip() for i in params_list
                                 if not i[0] in skip_params])
            if 'dict of keyworded arguments' in params_:
                import pydb
                pydb.debugger()
            initdoc += "\n\n:Parameters:\n" + _indent(params_)

        if suffix != "":
            initdoc += "\n\n" + suffix

        initdoc = handleDocString(initdoc)

        # Finally assign generated doc to the constructor
        lcl['__init__'].__doc__ = initdoc

    docs = [ handleDocString(lcl['__doc__']) ]

    # Optionally populate the class documentation with it
    if __add_init2doc and initdoc != "":
        docs += [ rstUnderline('Constructor information for `%s` class' % name,
                               rst_lvlmarkup[2]),
                  initdoc ]

    # Add information about the states if available
    if lcl.has_key('_statesdoc'):
        docs += ['.. note::\n  Available state variables:',
                 _indent(handleDocString(item._statesdoc))]

    if len(args):
        if len(args) > 1:
            bc_intro = '  Please refer to the documentation of the base ' \
                       'classes for more information:'
        else:
            bc_intro = '  Please refer to the documentation of the base ' \
                       'class for more information:'

        docs += ['\n.. seealso::',
                 bc_intro,
                 '  ' + ',\n  '.join([':class:`~%s.%s`' % (i.__module__,
                                                           i.__name__)
                                                              for i in args])
                ]

    itemdoc = '\n\n'.join(docs)
    # remove some bogus new lines -- never 3 empty lines in doc are useful
    result = re.sub("\s*\n\s*\n\s*\n", "\n\n", itemdoc)

    return itemdoc


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
    markup_strip = re.compile('^@[lrc]')
    col_width = [ max( [len(markup_strip.sub('', x))
                        for x in column] ) for column in atable.T ]
    string = ""
    for i, table_ in enumerate(table):
        string_ = ""
        for j, item in enumerate(table_):
            item = str(item)
            if item.startswith('@'):
                align = item[1]
                item = item[2:]
                if not align in ['l', 'r', 'c']:
                    raise ValueError, 'Unknown alignment %s. Known are l,r,c'
            else:
                align = 'c'

            NspacesL = ceil((col_width[j] - len(item))/2.0)
            NspacesR = col_width[j] - NspacesL - len(item)

            if align == 'c':
                pass
            elif align == 'l':
                NspacesL, NspacesR = 0, NspacesL + NspacesR
            elif align == 'r':
                NspacesL, NspacesR = NspacesL + NspacesR, 0
            else:
                raise RuntimeError, 'Should not get here with align=%s' % align

            string_ += "%%%ds%%s%%%ds " \
                       % (NspacesL, NspacesR) % ('', item, '')
        string += string_.rstrip() + '\n'
    out.write(string)

    if print2string:
        value = out.getvalue()
        out.close()
        return value

    pass

