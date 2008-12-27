#!/usr/bin/env python
# Attempt to generate templates for module reference with Sphinx

import os
import re
import mvpa

# blacklist some pieces that should not show up
exclude_list = ['mvpa.misc.copy']


modref_path = os.path.join('build', 'docsrc', 'modref')
if not os.path.exists(modref_path):
    os.mkdir(modref_path)

# only separating first two levels
rst_section_levels = ['*', '=', '-', '~', '^']

def getObjectName(line):
    name = line.split()[1].split('(')[0].strip()
    # in case we have classes which are niot derived from object
    # ie. old style classes
    return name.rstrip(':')


def parseModule(uri):
    filename = re.sub('\.', os.path.sep, uri)

    # get file uri
    if os.path.exists(filename + '.py'):
        filename += '.py'
    elif  os.path.exists(os.path.join(filename, '__init__.py')):
        filename = os.path.join(filename, '__init__.py')
    else:
        # nothing that we could handle here.
        return ([],[])

    f = open(filename)

    functions = []
    classes = []

    for line in f:
        if line.startswith('def ') and line.count('('):
            # exclude private stuff
            name = getObjectName(line)
            if not name.startswith('_'):
                functions.append(name)
        elif line.startswith('class '):
            # exclude private stuff
            name = getObjectName(line)
            if not name.startswith('_'):
                classes.append(name)
        else:
            pass

    f.close()

    functions.sort()
    classes.sort()

    return functions, classes


def writeAPIDocTemplate(uri):
    # get the names of all classes and functions
    functions, classes = parseModule(uri)

    # do nothing if there is nothing to do
    if not len(functions) and not len(classes):
        return

    tf = open(os.path.join(modref_path, uri + '.txt'), 'w')

    ad = '.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n'
    title = ':mod:`' + uri + '`'
    ad += title + '\n' + rst_section_levels[1] * len(title)

    ad += '\n.. automodule:: ' + uri + '\n'
    ad += '\n.. currentmodule:: ' + uri + '\n'

    ad += '\n\nThe comprehensive API documentation for this module, including\n' \
          'all technical details, is available in the Epydoc-generated `API\n' \
          'reference for %s`_ (for developers).\n\n' % uri
    ad += '.. _API reference for %s: ../api/%s-module.html\n\n' % (uri, uri)

    multi_class = len(classes) > 1
    multi_fx = len(functions) > 1
    if multi_class:
        ad += '\n' + 'Classes' + '\n' + rst_section_levels[2] * 7 + '\n'
    elif len(classes) and multi_fx:
        ad += '\n' + 'Class' + '\n' + rst_section_levels[2] * 5 + '\n'

    for c in classes:
        ad += '\n:class:`' + c + '`\n' \
              + rst_section_levels[multi_class + 2 ] * (len(c)+9) + '\n\n'

        ad += '\n.. autoclass:: ' + c + '\n'

        # must NOT exclude from index to keep cross-refs working
        ad += '  :members:\n' \
              '  :undoc-members:\n' \
              '  :show-inheritance:\n'
        #      '  :noindex:\n\n'

        # place api link
        # Reference to base classes might go away, since we have
        # enhancedDocString which does a superb job
        ad += '.. seealso::\n\n' \
              '  Derived classes might provide additional methods via their ' \
              '  base classes. Please refer to the list of base classes ' \
              '  (if it exists) at the begining of the ' \
              '  :class:`~' + c + '` documentation.\n\n' \
              '  Full API documentation of ' \
              '`%s in module %s`_.\n\n' % (c, uri)
        ad += '.. _%s in module %s: ../api/%s.%s-class.html\n\n' % (c, uri, uri, c)

    if multi_fx:
        ad += '\n' + 'Functions' + '\n' + rst_section_levels[2] * 9 + '\n\n'
    elif len(functions) and multi_class:
        ad += '\n' + 'Function' + '\n' + rst_section_levels[2] * 8 + '\n\n'

    for f in functions:
        # must NOT exclude from index to keep cross-refs working
        ad += '\n.. autofunction:: ' + uri + '.' + f + '\n\n'

        # place api link
        ad += '.. seealso::\n\n' \
              '  Full API documentation of ' \
              '`%s() in module %s`_.\n\n' % (f, uri)
        ad += '.. _%s() in module %s: ../api/%s-module.html#%s\n\n' % (f, uri, uri, f)
    tf.write(ad)
    tf.close()


root_path = mvpa.__path__[0]

# compose list of modules
modules = []

# raw directory parsing
for dirpath, dirnames, filenames in os.walk(mvpa.__path__[0]):
    # determine the importable location of the module
    module_uri = re.sub(os.path.sep,
                        '.',
                        re.sub(root_path,
                               'mvpa',
                               dirpath))

    # no unittests in docs
    if module_uri.startswith('mvpa.tests'):
        continue

    # no private module
    if not module_uri.count('._'):
        modules.append(module_uri)

    for filename in filenames:
        # XXX maybe check for extenstions as well?
        # not private stuff
        if not filename.endswith('.py') or filename.startswith('_'):
            continue

        modules.append('.'.join([module_uri, filename[:-3]]))

# write the list
for m in modules:
    if not m in exclude_list:
        writeAPIDocTemplate(m)
